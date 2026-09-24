"""GlucoTrace public import name.

``import glucotrace`` exposes the same API as the original ``glucofm``
package, which remains available for backward compatibility.
"""

from glucofm import *  # noqa: F401,F403
from glucofm import __all__, __version__  # noqa: F401
