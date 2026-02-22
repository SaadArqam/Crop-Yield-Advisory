"""Compatibility shim for imghdr.what using Pillow.

This provides a minimal replacement for the stdlib `imghdr.what` so that
`import imghdr` succeeds in environments where the stdlib module is missing
or unavailable (some sandboxed deploy environments).

It implements what(filename_or_file, h=None) which returns a short format name
(e.g. 'jpeg', 'png', 'gif') or None if unknown.
"""
from typing import Optional

try:
    from PIL import Image
    from io import BytesIO

    def what(file, h: Optional[bytes] = None) -> Optional[str]:
        """Return the image type for file or byte header h.

        Accepts either a filename/path or a file-like object. If `h` is provided
        it will try to detect type from the bytes buffer.
        """
        try:
            if h is not None:
                buf = BytesIO(h)
                img = Image.open(buf)
            else:
                # Accept Path-like, filename or file-like object
                img = Image.open(file)
            fmt = getattr(img, 'format', None)
            if not fmt:
                return None
            fmt = fmt.lower()
            # Normalize common names
            mapping = {
                'jpeg': 'jpeg',
                'png': 'png',
                'gif': 'gif',
                'bmp': 'bmp',
                'tiff': 'tiff',
                'webp': 'webp'
            }
            return mapping.get(fmt, fmt)
        except Exception:
            return None

except Exception:
    # Pillow not available or import failed: provide a safe fallback implementation
    def what(file, h: Optional[bytes] = None) -> Optional[str]:
        return None
