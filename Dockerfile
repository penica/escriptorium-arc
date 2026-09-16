ARG ARC_RUNTIME_IMAGE
FROM node@sha256:4517380049fc3c9aacceae7764fcf3500354b0ac8a47e4afb35b5bbeb75b9498 AS frontend
WORKDIR /build
COPY source/front/ /build/
ENV NODE_OPTIONS=--max-old-space-size=2048
RUN npm ci --no-audit && npm run production
FROM ${ARC_RUNTIME_IMAGE}
ARG UPSTREAM_COMMIT
USER root
RUN python -m pip freeze > /opt/arc-runtime-constraints.txt && python -m pip install --no-cache-dir -c /opt/arc-runtime-constraints.txt django-webpack-loader==3.1.1 'drf-spectacular[sidecar]==0.29.0' && python -m pip check
RUN rm -rf /usr/src/app/apps /usr/src/app/escriptorium /usr/src/app/front /usr/src/app/locale /usr/src/app/homepage /usr/src/app/contributors && rm -f /usr/src/app/arc_integration.py
COPY source/app/ /usr/src/app/
COPY --from=frontend /build/dist /usr/src/app/front
COPY extensions/ /usr/src/app/
RUN mkdir -p /usr/src/app/videm-seg-v4
ENV VIDEM_V4_ENABLED=0

RUN apt-get update -q && apt-get install -y --no-install-recommends gettext curl patch && apt-get clean && rm -rf /var/lib/apt/lists/*
COPY patches/ /opt/arc-build/
COPY licenses/ /opt/arc-build/licenses/
RUN cd /usr/local/lib/python3.12/site-packages/kraken/lib && patch --batch --forward --fuzz=0 -p1 < /opt/arc-build/kraken-segmentation.patch
RUN python /opt/arc-build/app.py && python -m compileall -q /usr/src/app/apps /usr/src/app/escriptorium /usr/src/app/videm_training.py && python manage.py compilemessages && python -m pip freeze > /opt/arc-build/installed.lock.txt
ENV VERSION_DATE=develop-${UPSTREAM_COMMIT} FRONTEND_DIR=/usr/src/app/front TORCH_COMPILE_DISABLE=1
COPY scripts/smoke-image.py /opt/arc-build/smoke-image.py
COPY scripts/check-routing.py /opt/arc-build/check-routing.py
RUN PYTHONPATH=/usr/src/app:/usr/src/app/apps python /opt/arc-build/check-routing.py --celery
RUN PYTHONPATH=/usr/src/app:/usr/src/app/apps python /opt/arc-build/smoke-image.py
LABEL org.opencontainers.image.revision="${UPSTREAM_COMMIT}" org.opencontainers.image.source="https://github.com/penica/escriptorium-arc" org.opencontainers.image.upstream="https://gitlab.com/scripta/escriptorium.git"
