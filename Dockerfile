FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        gcc \
        libc-dev \
        slurm-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY advQMcalc_test/ /srv/advQMcalc_test/
COPY app/ /srv/app/
COPY extra/slurm.conf /srv/app/slurm.conf

# tornado>=6.5 added a symlink-boundary security check
# (allowed_symlink_directory / _resolve_symlink_target) that several of
# voila's own static-file handler classes were never updated for: it's
# pinned at handler-init time to a single directory (sometimes a
# placeholder that doesn't even exist) and never reconciled with the
# multi-directory search these handlers actually perform, so any asset
# found outside that one directory 403s as "not in root static directory"
# (or even 500s for the labextensions handler, which doesn't set the
# attribute at all). This breaks voila.js, the template CSS/JS, and the
# ipywidgets labextension bundles — i.e. the whole UI never renders past
# "Executing N of N". Pinning tornado below 6.5 restores the older,
# permissive behavior these handlers were actually written against.
#
# ipython>=9 also removed IPython.core.pylabtools.backend2gui, which
# voila's render path imports — pinning below 9 avoids that ImportError.
RUN pip install --no-cache-dir \
        /srv/advQMcalc_test \
        "tornado<6.5" \
        "ipython<9" \
        voila \
        ipywidgets \
        ipykernel \
        pandas

RUN pip install --no-cache-dir cif2cell

RUN chmod -R 777 /srv

ENV SLURM_CONF=/srv/app/slurm.conf

# --template=classic: the default "lab" template ships with no compiled
# static/ bundle at all in a plain pip install (needs a
# `jupyter labextension build` this image never runs) — unrelated to the
# tornado issue above, just an incomplete package. "classic" is the
# complete, self-contained template that actually has one.
#
# Voila renders the page as a long-lived chunked HTTP response and relies on
# periodic <script>window.voila_heartbeat()</script> pings (every
# http_keep_alive_timeout seconds) to keep any proxy in front of it from
# treating the connection as idle and cutting it — which shows up in the
# browser as ERR_INCOMPLETE_CHUNKED_ENCODING. The 10s default was apparently
# too infrequent for the proxy fronting this app, so send heartbeats every 2s.
CMD ["voila", "--port=8866", "--Voila.ip=0.0.0.0", "--no-browser", "--autoreload=False", "--template=classic", "--VoilaConfiguration.http_keep_alive_timeout=2", "app/app.ipynb"]
