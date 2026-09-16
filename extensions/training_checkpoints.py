"""Full-state checkpoint IO for trusted, local/shared-filesystem training runs."""
import os
import uuid
from pathlib import Path
from lightning.fabric.plugins.io import TorchCheckpointIO
from training_safety import assert_training_lock
from training_policy import ensure_shared_directory


class AtomicCheckpointIO(TorchCheckpointIO):
    """Publish a checkpoint only after serialization and fsync have succeeded."""

    def save_checkpoint(self, checkpoint, path, storage_options=None):
        if storage_options is not None:
            raise TypeError('Training checkpoint storage_options are not supported')
        if '://' in str(path):
            raise ValueError('Training checkpoints require a filesystem path')
        target = Path(path)
        ensure_shared_directory(target.parent)
        temporary = target.with_name('.' + target.name + '.' + uuid.uuid4().hex + '.pending')
        assert_training_lock()
        try:
            super().save_checkpoint(checkpoint, temporary)
            with temporary.open('rb') as stream:
                os.fsync(stream.fileno())
            assert_training_lock()
            os.replace(temporary, target)
            descriptor = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            temporary.unlink(missing_ok=True)
