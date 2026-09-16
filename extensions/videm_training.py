from core.tasks import (os, User, settings, apps, slugify, send_event, _, np, logger, make_segmentation_training_data, _to_ptl_device, BLLASegmentationTrainingDataConfig, BLLASegmentationTrainingConfig, BLLASegmentationDataModule, BLLASegmentationModel, KrakenTrainer, ModelCheckpoint, FrontendFeedback, DidNotConverge, convert_models, shutil)
def _videm_v4_segtrain(model_pk=None, part_pks=[], document_pk=None, task_group_pk=None, user_pk=None, **kwargs):
    # # Note hack to circumvent AssertionError: daemonic processes are not allowed to have children
    from multiprocessing import current_process
    current_process().daemon = False

    if user_pk:
        try:
            user = User.objects.get(pk=user_pk)
            # If quotas are enforced, assert that the user still has free CPU minutes, GPU minutes and disk storage
            if not settings.DISABLE_QUOTAS:
                if user.cpu_minutes_limit() is not None:
                    assert user.has_free_cpu_minutes(), f"User {user.id} doesn't have any CPU minutes left"
                if user.gpu_minutes_limit() is not None:
                    assert user.has_free_gpu_minutes(), f"User {user.id} doesn't have any GPU minutes left"
                if user.disk_storage_limit() is not None:
                    assert user.has_free_disk_storage(), f"User {user.id} doesn't have any disk storage left"
        except User.DoesNotExist:
            user = None
    else:
        user = None

    Document = apps.get_model('core', 'Document')
    DocumentPart = apps.get_model('core', 'DocumentPart')
    OcrModel = apps.get_model('core', 'OcrModel')

    model = OcrModel.objects.get(pk=model_pk)

    try:
        load = model.file.path
    except ValueError:  # model is empty
        load = '/usr/src/app/videm-seg-v4/start-v3-regions.safetensors'
        model.file = model.file.field.upload_to(model, slugify(model.name) + '.safetensors')

    model_dir = os.path.join(settings.MEDIA_ROOT, os.path.split(model.file.path)[0])

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if load:
        import json

        from safetensors import SafetensorError, safe_open
        try:
            with safe_open(load, framework="pt") as f:
                raw_meta = f.metadata()
            kraken_meta = json.loads(raw_meta.get('kraken_meta')) if raw_meta else {}
        except (ValueError, TypeError, json.JSONDecodeError, SafetensorError):
            kraken_meta = {}
        if any(v.get('_model') == 'DFINEModel' for v in kraken_meta.values() if isinstance(v, dict)):
            send_event('document', document_pk, "training:error", {"id": model.pk})
            if user:
                user.notify(_("D-FINE model fine-tuning is not supported at the moment."),
                            id="training-dfine-unsupported", level='warning')
            model.delete()
            raise NotImplementedError(_("D-FINE model fine-tuning is not supported at the moment."))

    try:
        model.training = True
        model.save()
        send_event('document', document_pk, "training:start", {
            "id": model.pk,
        })
        qs = DocumentPart.objects.filter(pk__in=part_pks).prefetch_related('lines')

        ground_truth = list(qs)
        if ground_truth[0].document.line_offset == Document.LINE_OFFSET_TOPLINE:
            topline = True
        elif ground_truth[0].document.line_offset == Document.LINE_OFFSET_CENTERLINE:
            topline = None
        else:
            topline = False

        np.random.default_rng(241960353267317949653744176059648850006).shuffle(ground_truth)
        partition = max(1, int(len(ground_truth) / 10))

        from videm_seg_v4 import CONFIG, seed, configure, EntryMetrics
        assert document_pk == CONFIG['document']
        assert set(part_pks) == set(CONFIG['train'] + CONFIG['validation'])
        assert not set(part_pks) & set(CONFIG['test'])
        seed()
        training_data = make_segmentation_training_data(qs.filter(pk__in=CONFIG['train']).order_by('pk'))
        evaluation_data = make_segmentation_training_data(qs.filter(pk__in=CONFIG['validation']).order_by('pk'))

        accelerator, device = _to_ptl_device(getattr(settings, 'KRAKEN_TRAINING_DEVICE', 'cpu'))

        LOAD_THREADS = getattr(settings, 'KRAKEN_TRAINING_LOAD_THREADS', 0)
        AMP_MODE = getattr(settings, 'KRAKEN_TRAINING_PRECISION', '32')

        logger.info(f'Starting segmentation training on {accelerator}/{device} '
                    f'(precision: {AMP_MODE}, workers: {LOAD_THREADS}) with '
                    f'{len(training_data)} files')

        seg_data_config = BLLASegmentationTrainingDataConfig(
            training_data=training_data,
            evaluation_data=evaluation_data,
            format_type=None,
            line_class_mapping={}, region_class_mapping={'BaptismEntry':2}, augment=True,
            num_workers=LOAD_THREADS,
        )
        seg_train_config = BLLASegmentationTrainingConfig(
            resize='fail',
            topline=topline,
            load_hyper_parameters=False,
            lrate=0.0001, optimizer='AdamW', schedule='cosine', epochs=40, cos_t_max=40, cos_min_lr=0.00001, warmup=20,
        )
        seg_dm = BLLASegmentationDataModule(seg_data_config)
        if load:
            kraken_model = BLLASegmentationModel.load_from_weights(load, seg_train_config)
        else:
            kraken_model = BLLASegmentationModel(seg_train_config)

        configure(kraken_model, seg_dm)
        trainer = KrakenTrainer(accelerator=accelerator,
                                devices=device,
                                max_epochs=40,
                                min_epochs=1,
                                gradient_clip_val=1.0,
                                precision=AMP_MODE,
                                enable_summary=False,
                                enable_progress_bar=False,
                                num_sanity_val_steps=-1,
                                val_check_interval=1.0,
                                callbacks=[EntryMetrics(), ModelCheckpoint(dirpath=model_dir, monitor='val_entry_score', mode='max', save_top_k=1, save_last=True, filename='region_{epoch:02d}-{val_entry_score:.5f}'), FrontendFeedback(model, model_dir, document_pk)])

        trainer.fit(kraken_model, seg_dm)

        # best checkpoint path from lightning's ModelCheckpoint callback
        best_path = getattr(getattr(trainer, 'checkpoint_callback', None), 'best_model_path', None)
        best_score = getattr(getattr(trainer, 'checkpoint_callback', None), 'best_model_score', None)

        if not best_path:
            logger.info(f'Model {os.path.split(model.file.path)[0]} did not improve.')
            raise DidNotConverge

        try:
            best_score_val = float(best_score) if best_score is not None else 0.0
            logger.info(f'Converting best model {best_path} (accuracy: {best_score_val}) to {model.file.path}.')
            convert_models([best_path], model.file.path)
            from kraken.models import load_models, write_safetensors
            from entry_extraction import CONFIG as extractor_config
            import hashlib
            trained = load_models(model.file.path)[0]
            def weight_digest(net):
                h = hashlib.sha256()
                for key, value in sorted(net.nn.state_dict().items()):
                    h.update(key.encode()); h.update(value.detach().cpu().numpy().tobytes())
                return h.hexdigest()
            before = weight_digest(trained)
            trained.user_metadata['videm_entry_extraction'] = {**extractor_config, 'version': 3}
            trained.user_metadata['videm_training_round'] = 4
            write_safetensors([trained], model.file.path)
            assert weight_digest(load_models(model.file.path)[0]) == before
            import re
            from pathlib import Path
            history = [json.loads(row) for row in Path('/usr/src/app/videm-seg-v4/history.jsonl').read_text().splitlines()]
            selected_epoch = int(re.search(r'epoch[=_](\d+)', best_path).group(1)) + 1
            selected = next(row for row in history if row['epoch'] == selected_epoch)
            model.training_accuracy = selected['f1']
            Path('/usr/src/app/videm-seg-v4/export-proof.json').write_text(json.dumps({'selected_epoch': selected_epoch, 'selected_checkpoint': best_path, 'validation_entry_f1': selected['f1'], 'composite_selection_score': best_score_val, 'weights_unchanged_by_metadata_restore': True, 'weight_digest': before}, indent=2))
        except FileNotFoundError:
            logger.info(f'Model {os.path.split(model.file.path)[0]} did not improve.')
            if user:
                user.notify(_("Training didn't get better results than base model!"),
                            id="seg-no-gain-error", level='warning')
            shutil.copy(load, model.file.path)

    except DidNotConverge:
        send_event('document', ground_truth[0].document.pk, "training:error", {
            "id": model.pk,
        })
        user.notify(_("The model did not converge, probably because of lack of data."),
                    id="training-warning", level='warning')
        model.delete()

    except Exception as e:
        send_event('document', document_pk, "training:error", {
            "id": model.pk,
        })
        if user:
            user.notify(_("Something went wrong during the segmenter training process!"),
                        id="training-error", level='danger')
        logger.exception(e)
        raise e
    else:
        model.file_size = model.file.size

        if user:
            user.notify(_("Training finished!"),
                        id="training-success",
                        level='success')
    finally:
        model.training = False
        model.save()

        send_event('document', document_pk, "training:done", {
            "id": model.pk,
        })
