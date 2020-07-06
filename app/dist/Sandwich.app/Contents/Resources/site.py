"""
Append module search paths for third-party packages to sys.path.

This is stripped down and customized for use in py2app applications
"""

import sys
# os is actually in the zip, so we need to do this here.
# we can't call it python24.zip because zlib is not a built-in module (!)
_libdir = '/lib/python' + sys.version[:3]
_parent = '/'.join(__file__.split('/')[:-1])
if not _parent.endswith(_libdir):
    _parent += _libdir
sys.path.append(_parent + '/site-packages.zip')

# Stuffit decompresses recursively by default, that can mess up py2app bundles,
# add the uncompressed site-packages to the path to compensate for that.
sys.path.append(_parent + '/site-packages')

USER_SITE=None

import os
try:
    basestring
except NameError:
    basestring = str

def makepath(*paths):
    dir = os.path.abspath(os.path.join(*paths))
    return dir, os.path.normcase(dir)

for m in sys.modules.values():
    f = getattr(m, '__file__', None)
    if isinstance(f, basestring) and os.path.exists(f):
        m.__file__ = os.path.abspath(m.__file__)
del m

# This ensures that the initial path provided by the interpreter contains
# only absolute pathnames, even if we're running from the build directory.
L = []
_dirs_in_sys_path = {}
dir = dircase = None  # sys.path may be empty at this point
for dir in sys.path:
    # Filter out duplicate paths (on case-insensitive file systems also
    # if they only differ in case); turn relative paths into absolute
    # paths.
    dir, dircase = makepath(dir)
    if not dircase in _dirs_in_sys_path:
        L.append(dir)
        _dirs_in_sys_path[dircase] = 1
sys.path[:] = L
del dir, dircase, L
_dirs_in_sys_path = None

def _init_pathinfo():
    global _dirs_in_sys_path
    _dirs_in_sys_path = d = {}
    for dir in sys.path:
        if dir and not os.path.isdir(dir):
            continue
        dir, dircase = makepath(dir)
        d[dircase] = 1

def addsitedir(sitedir):
    global _dirs_in_sys_path
    if _dirs_in_sys_path is None:
        _init_pathinfo()
        reset = 1
    else:
        reset = 0
    sitedir, sitedircase = makepath(sitedir)
    if not sitedircase in _dirs_in_sys_path:
        sys.path.append(sitedir)        # Add path component
    try:
        names = os.listdir(sitedir)
    except os.error:
        return
    names.sort()
    for name in names:
        if name[-4:] == os.extsep + "pth":
            addpackage(sitedir, name)
    if reset:
        _dirs_in_sys_path = None

def addpackage(sitedir, name):
    global _dirs_in_sys_path
    if _dirs_in_sys_path is None:
        _init_pathinfo()
        reset = 1
    else:
        reset = 0
    fullname = os.path.join(sitedir, name)
    try:
        with open(fullname) as f:
            while 1:
                dir = f.readline()
                if not dir:
                    break
                if dir[0] == '#':
                    continue
                if dir.startswith("import"):
                    exec(dir)
                    continue
                if dir[-1] == '\n':
                    dir = dir[:-1]
                dir, dircase = makepath(sitedir, dir)
                if not dircase in _dirs_in_sys_path and os.path.exists(dir):
                    sys.path.append(dir)
                    _dirs_in_sys_path[dircase] = 1
    except IOError:
        return
    if reset:
        _dirs_in_sys_path = None


#sys.setdefaultencoding('utf-8')

#
# Run custom site specific code, if available.
#
try:
    import sitecustomize
except ImportError:
    pass

#
# Remove sys.setdefaultencoding() so that users cannot change the
# encoding after initialization.  The test for presence is needed when
# this module is run as a script, because this code is executed twice.
#
if hasattr(sys, "setdefaultencoding"):
    del sys.setdefaultencoding
