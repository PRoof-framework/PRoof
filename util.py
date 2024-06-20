import os
import logging
from pathlib import Path
import sys
import tempfile

log = logging.getLogger(__name__)


TMPDIR = Path(os.getenv('PROOF_TMPDIR', str(Path(tempfile.gettempdir()) / 'proof')))
try:
    Path(TMPDIR).mkdir(parents=True, exist_ok=True)
    assert os.access(TMPDIR, os.W_OK)
except Exception as e:
    log.error('TMPDIR %s is not writeable (%s), you can change it via DNNROOF_TMPDIR env var', TMPDIR, e)
    sys.exit(1)
