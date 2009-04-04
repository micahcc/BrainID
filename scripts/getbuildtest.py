#!env python
import os
from os.path import join, split, splitext 
import shutil
import sys
from urllib import urlretrieve
import tarfile
from optparse import OptionParser
import time


# some user modifiable constants
CMAKE_URL="http://www.cmake.org/files/v2.6/cmake-2.6.2.tar.gz"
DYSII_URL="http://www.indii.org/files/dysii/releases/dysii-1.4.0.tar.gz"
GSL_URL="ftp://ftp.gnu.org/gnu/gsl/gsl-1.9.tar.gz"
#BOOST_URL="http://downloads.sourceforge.net/boost/boost_1_38_0.tar.gz?use_mirror=voxel"
BOOST_URL="http://voxel.dl.sourceforge.net/sourceforge/boost/boost_1_38_0.tar.gz"
BINDINGS_URL="http://mathema.tician.de/news.tiker.net/download/software/boost-numeric-bindings/boost-numeric-bindings-20081116.tar.gz"
ITK_URL="http://voxel.dl.sourceforge.net/sourceforge/itk/InsightToolkit-3.12.0.tar.gz"
OPENMPI_URL="http://www.open-mpi.org/software/ompi/v1.3/downloads/openmpi-1.3.1.tar.gz"
LAPACK_URL="http://www.netlib.org/lapack/lapack-3.1.1.tgz"


def getep(basedir, name, url):
    archive_file = split(url)[1]
    archive_path = join(basedir, archive_file)

    src_dir = join(basedir, splitext(archive_file)[0])
    if splitext(src_dir)[1] == ".tar":
        src_dir = splitext(src_dir)[0]
    
    if not os.path.exists(src_dir):
        if not os.path.exists(archive_path):
            print "downloading %s" % name
            urlretrieve(url, archive_path, progress)
        tarobj = tarfile.open(archive_path, 'r:gz')
        tarobj.extractall(basedir)
   
    if os.path.exists("%s.patch" %name) and os.system("patch --dry-run -Np2 -d %s < %s" %(src_dir, "%s.patch" % name)) == 0:
        print "Patching %s" % name
        os.system("patch -Np2 -d %s < %s" % (src_dir, "%s.patch" % name))
    return src_dir

#defstrings should be a tuple of string arguments to pass to configure
def buildboost(basedir, instdir, name, url, defstrings = ""):
    topdir = os.getcwd()
    src_dir = getep(basedir, name, url)
    install_dir = join(instdir, name)
    
    print "building %s" % name
    os.chdir(src_dir)
    if os.system("make -j%i" % ncpus()) != 0:
        if  os.system("./configure --prefix=%s %s" % (install_dir, " ".join(defstrings))) != 0:
            print "%s configuration failed" % name
            sys.exit()
        os.system("echo using mpi \; >> %s/user-config.jam" % src_dir)
        if os.system("make -j%i" % ncpus()) != 0:
            print "build in %s failed" % src_dir
            sys.exit()
    
    if os.system("make install") != 0:
        print "make install in %s failed" % src_dir
        sys.exit()
    print "build of %s completed" % name
    
    try:
        shutil.rmtree(join(install_dir, "include", "boost"))
    except os.error:
        pass
    shutil.move(join(install_dir, "include","boost-1_38","boost"), \
                join(install_dir, "include", "boost"))
    shutil.rmtree(join(install_dir, "include", "boost-1_38"))

    os.chdir(join(install_dir, "lib"));

    liblist = [];
    for file in os.listdir("./"):
        try:
            os.readlink(file)
            print "Success: %s" % file
        except os.error:
            print "Error: %s" % file
            liblist.append(file)

    print liblist
    for file in liblist:
        print file
        print splitext(file)
        print file.split("-")
        if splitext(file)[1] == ".a":
            print "Symlinking %s -> %s" % (file.split("-")[0] + ".a", file)
            try: 
                os.symlink(file, file.split("-")[0] +".a");
            except os.error:
                pass
        else:
            print "Symlinking %s -> %s" % (file.split("-")[0] + ".so", file)
            try: 
                os.symlink(file, file.split("-")[0] + ".so");
            except os.error:
                pass

    os.chdir(topdir)
    return install_dir

#defstrings should be a tuple of string arguments to pass to configure
def confmakeinst(basedir, instdir, name, url, defstrings = ""):
    topdir = os.getcwd()
    src_dir = getep(basedir, name, url)
    install_dir = join(instdir, name)
    
    print "building %s" % name
    os.chdir(src_dir)
    if os.system("make -j%i" % ncpus()) != 0:
        if  os.system("./configure --prefix=%s %s" % (install_dir, " ".join(defstrings))) != 0:
            print "%s configuration failed" % name
            sys.exit()
        if os.system("make -j%i" % ncpus()) != 0:
            print "build in %s failed" % src_dir
            sys.exit()
    
    if os.system("make install") != 0:
        print "make install in %s failed" % src_dir
        sys.exit()
    os.chdir(topdir)
    print "build of %s completed" % name
    return install_dir

#defstrings should be a tuple of extra arguments to give to cmake
def cmakeinst(basedir, instdir, name, url, defstrings = ""):
    topdir = os.getcwd()
    src_dir = getep(basedir,name,url)
    install_dir = join(instdir, name)
    
    print "Building %s" % name
    build_dir = join(basedir, "%s-build" % name)
    try:
        os.makedirs(build_dir)
    except os.error:
        print "Directory %s exists, using it" % build_dir
    
    os.chdir(build_dir)
    if os.system("cmake %s -DCMAKE_INSTALL_PREFIX=%s %s" % (src_dir, install_dir, " ".join(defstrings))) != 0:
        print "%s configuration in %s failed" % (name, build_dir)
        sys.exit()
    if os.system("make -j%i" % ncpus()) != 0:
        print "build in %s failed" % build_dir
        sys.exit()
    if os.system("make install") != 0:
        print "build in %s failed" % build_dir
        sys.exit()
    os.chdir(topdir)
    print "Build of %s Completed" % name
    return install_dir

def ncpus():
    try:
        if hasattr(os, "sysconf"): 
            if os.sysconf_names.has_key("SC_NPROCESSORS_ONLN"): # Linux and Unix 
                n = os.sysconf("SC_NPROCESSORS_ONLN")  
                if isinstance(n, int) and n > 0: 
                    return n 
            else:  # MacOS X 
                return int(os.popen2("sysctl -n hw.ncpu")[1].read()) 
    except: 
        return int(os.popen2("sysctl -n hw.ncpu")[1].read().strip()) 


def progress(blocks, blocksize, total):
    ESC = chr(27);
    sys.stdout.write(ESC + '[2K' + ESC+'[G')
    text = " %i of %i bytes (%.0f percent) complete" % (blocks*blocksize, total, float(blocks*blocksize*100)/total) 
    sys.stdout.write(text + '\x08'*len(text))
    sys.stdout.flush()


topdir = os.getcwd()
srcpath = split(sys.path[0])[0]

parser = OptionParser()
parser.add_option("-d", "--depdir", dest="depdir",
                  default="BrainID-Dependencies-Build",
                  help="name of directory for dependencies")
parser.add_option("-u", "--update",
                  action="store_true", dest="update", default=False,
                  help="run svn update first")
parser.add_option("-i", "--depinstall",
                  action="store_true", dest="depprefix", default="%s/BrainID-Dependencies-Install" % topdir,
                  help="The location to put the final <depname>/include, <depname>/lib, <depname>/bin ")
parser.add_option("-p", "--prefix",
                  action="store_true", dest="prefix", default="%s" % join(srcpath, "install"),
                  help="The location to put the final include, lib, bin....dirs")
(options, args) = parser.parse_args()

PROFILE_OUT="%s/prof.sh" % options.depdir

depdir = join(topdir, options.depdir)
depprefix = join(topdir, options.depprefix)

platform = os.uname()[0]
if platform == 'Darwin':
        APPLE_ARCH = os.uname()[4]
        print "Detected platform Darwin, enabling universal binaries"
        CMAKE_APPLE_UNIV = "-D CMAKE_OSX_ARCHITECTURES=i386;ppc -D CMAKE_TRY_COMPILE_OSX_ARCHITECTURES=" + APPLE_ARCH


# make directories if they do not exist
try:
    os.makedirs(options.depdir)
except os.error:
    print "Directory " + options.depdir + " may overwrite."
    print "3"
    time.sleep(1);
    print "2"
    time.sleep(1);
    print "1"
    time.sleep(1);


##########################
# CMake
##########################
cmake_install_dir = cmakeinst(depdir, depprefix, "cmake", CMAKE_URL)
os.environ["PATH"] = join(cmake_install_dir, "bin") + ":"+ os.environ["PATH"]

###########################
# gsl
###########################
gsl_install_dir = confmakeinst(depdir, depprefix, "gsl", GSL_URL)

############################
# Boost 
############################
boost_install_dir = buildboost(depdir, depprefix, "boost", BOOST_URL, ("-with-libraries=serialization,mpi", "") )

############################
# Boost Numeric Bindings
############################
boost_numeric_bindings_archive_file = split(BINDINGS_URL)[1]
boost_numeric_bindings_archive_path = join(depdir, boost_numeric_bindings_archive_file)
boost_numeric_bindings_src_dir = join(depdir, (splitext(splitext(boost_numeric_bindings_archive_file)[0])[0]))
if not os.path.exists(boost_numeric_bindings_archive_path):
    print "Downloading Boost Numeric Bindings"
    urlretrieve(BINDINGS_URL, boost_numeric_bindings_archive_path, progress)
tarobj = tarfile.open(boost_numeric_bindings_archive_path, 'r:gz')
tarobj.extractall(depdir)

shutil.move(join(depdir, "boost-numeric-bindings", "boost", "numeric", "bindings"), \
            join(depprefix, "boost", "include", "boost", "numeric","bindings"))
os.chdir(topdir)

###########################
# openmpi
###########################
mpi_install_dir = confmakeinst(depdir, depprefix, "mpi", OPENMPI_URL)

###########################
# ITK
###########################
itk_install_dir = cmakeinst(depdir, depprefix, "itk", ITK_URL, ("-DBUILD_TESTING=OFF", "-DBUILD_EXAMPLES=OFF"))

###########################
# LAPACK
###########################
lapack_src_dir = getep(depdir, "lapack", LAPACK_URL)
os.chdir(lapack_src_dir)
os.system("make -j%i blaslib" % ncpus())
os.system("make -j%i all" % ncpus())
#for files in os.listdir("./"):
#    if splitext(files)[1] == ".a":
#        os.

#os.path.
os.chdir(topdir)

###########################
# dysii
###########################
dysii_install_dir = cmakeinst(depdir, depprefix, "dysii", DYSII_URL, ("-DGSL=%s" % gsl_install_dir, \
            "-DMPI=%s" % mpi_install_dir, "-DBOOST=%s" % boost_install_dir))

###########################
# brainid
###########################
os.chdir(srcpath)
brainid_build_dir = join(srcpath, "build")
brainid_install_dir = options.prefix
try:
    os.makedirs(brainid_build_dir)
except os.error:
    print "Directory %s exists, using it" %brainid_build_dir

try:
    os.makedirs(brainid_install_dir)
except os.error:
    print "Directory %s exists, using it" %brainid_install_dir

os.chdir(brainid_build_dir)
if os.system("cmake %s -DITK_DIR=%s -Ddysii_INCLUDE_DIRS=%s -Ddysii_LIBRARY_DIRS=%s -DCMAKE_INSTALL_PREFIX=%s"  % \
            (srcpath, join(itk_install_dir, "lib", "InsightToolkit"), \
            join(dysii_install_dir, "include"),  \
            join(dysii_install_dir, "lib"), brainid_install_dir)) != 0:
    print "dysii configuration in %s failed" % dysii_build_dir
    sys.exit()

if os.system("make -j%i" % ncpus()) != 0:
    print "build in %s failed" % brainid_build_dir
    sys.exit()

if os.system("make install") != 0:
    print "install from %s failed" % brainid_build_dir
    sys.exit()

os.chdir(topdir)
print "Build of brainid Completed"

