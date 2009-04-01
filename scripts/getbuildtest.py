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
BOOST_URL="http://downloads.sourceforge.net/boost/boost_1_38_0.tar.gz?use_mirror=voxel"
BINDINGS_URL="http://mathema.tician.de/news.tiker.net/download/software/boost-numeric-bindings/boost-numeric-bindings-20081116.tar.gz"
ITK_URL="http://voxel.dl.sourceforge.net/sourceforge/itk/InsightToolkit-3.12.0.tar.gz"
OPENMPI_URL="http://www.open-mpi.org/software/ompi/v1.3/downloads/openmpi-1.3.1.tar.gz"

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
                  default="BrainID-Dependencies",
                  help="name of directory for dependencies")
parser.add_option("-u", "--update",
                  action="store_true", dest="update", default=False,
                  help="run svn update first")
parser.add_option("-p", "--prefix",
                  action="store_true", dest="prefix", default="%s/root" % topdir,
                  help="The location to put the final include, lib, bin....dirs")
(options, args) = parser.parse_args()

PROFILE_OUT="%s/prof.sh" % options.depdir

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

#	sys.exit()


##########################
# CMake
##########################
cmake_archive_file = split(CMAKE_URL)[1]
cmake_archive_path = join(options.depdir, cmake_archive_file)
cmake_src_dir = join("..", (splitext(splitext(cmake_archive_file)[0])[0]))
cmake_build_dir = join(options.depdir, "cmake-build")
if not os.path.isfile(cmake_archive_path):
    print "Downloading CMake"
    urlretrieve(CMAKE_URL, cmake_archive_path, progress)
tarobj = tarfile.open(cmake_archive_path, 'r:gz')
tarobj.extractall(options.depdir)
print "Building CMake"
try:
    os.makedirs(cmake_build_dir)
except os.error:
    print "Directory %s exists, using it" %cmake_build_dir

os.chdir(cmake_build_dir)
if os.system("cmake %s -DCMAKE_INSTALL_PREFIX=%s" % (cmake_src_dir, options.prefix)) != 0:
	print "cmake configuration in %s failed" % cmake_build_dir
	sys.exit()
if os.system("make -j%i" % ncpus()) != 0:
	print "build in %s failed" % cmake_build_dir
	sys.exit()
if os.system("make install") != 0:
	print "build in %s failed" % cmake_build_dir
	sys.exit()
os.chdir(topdir)
print "Build of CMake Completed"

###########################
# gsl
###########################
gsl_archive_file = split(GSL_URL)[1]
gsl_archive_path = join(options.depdir, gsl_archive_file)
gsl_src_dir = join(options.depdir, (splitext(splitext(gsl_archive_file)[0])[0]))
if not os.path.isfile(gsl_archive_path):
    print "downloading gsl"
    urlretrieve(GSL_URL, gsl_archive_path, progress)
tarobj = tarfile.open(gsl_archive_path, 'r:gz')
tarobj.extractall(options.depdir)
print "building gsl"
os.chdir(gsl_src_dir)
if os.system("./configure --prefix=%s" % join(options.prefix)) != 0:
    print "gsl configuration failed"
    sys.exit()
if os.system("make -j%i" % ncpus()) != 0:
    print "build in %s failed" % gsl_src_dir
    sys.exit()
if os.system("make install") != 0:
    print "make install in %s failed" % gsl_src_dir
    sys.exit()
os.chdir(topdir)
print "build of gsl completed"

############################
# Boost 
############################
#build serialization
boost_archive_file = split(BOOST_URL)[1]
boost_archive_file = boost_archive_file.partition("?")[0];
boost_archive_path = join(options.depdir, boost_archive_file)
boost_src_dir = join(options.depdir , (splitext(splitext(boost_archive_file)[0])[0]))
if not os.path.exists(boost_archive_path):
    print "Downloading Boost"
    urlretrieve(BOOST_URL, boost_archive_path, progress)
tarobj = tarfile.open(boost_archive_path, 'r:gz')
tarobj.extractall(options.depdir)
print "Building Boost Serialization Library"
os.chdir(boost_src_dir)
if os.system("./configure --with-libraries=serialization --prefix=%s" % options.prefix) != 0:
    print "boost configuration failed"
    sys.exit()
if os.system("make -j%i" % ncpus()) != 0:
    print "build in %s failed" % boost_src_dir
    sys.exit()
if os.system("make install") != 0:
    print "make install in %s failed" % boost_src_dir
    sys.exit()
#KLUDGE 
if os.path.exists(join(options.prefix, "include", "boost")):
    shutil.rmtree(join(options.prefix, "include", "boost"))
shutil.move(join(options.prefix, "include","boost-1_38","boost"), join(options.prefix, "include", "boost"))
shutil.rmtree(join(options.prefix, "include", "boost-1_38"))
os.chdir(topdir)

############################
# Boost Numeric Bindings
############################
boost_numeric_bindings_archive_file = split(BINDINGS_URL)[1]
boost_numeric_bindings_archive_path = join(options.depdir, boost_numeric_bindings_archive_file)
boost_numeric_bindings_src_dir = join(options.depdir, (splitext(splitext(boost_numeric_bindings_archive_file)[0])[0]))
if not os.path.exists(boost_numeric_bindings_archive_path):
    print "Downloading Boost Numeric Bindings"
    urlretrieve(BINDINGS_URL, boost_numeric_bindings_archive_path, progress)
tarobj = tarfile.open(boost_numeric_bindings_archive_path, 'r:gz')
tarobj.extractall(options.depdir)
if os.path.exists(join(options.prefix, "include", "boost", "numeric", "bindings")):
    shutil.rmtree(join(options.prefix, "include", "boost", "numeric", "bindings"))
shutil.move(join(options.depdir, "boost-numeric-bindings", "boost", "numeric", "bindings"), join(options.prefix, "include", "boost", "numeric","bindings"))
os.chdir(topdir)

###########################
# openmpi
###########################
mpi_archive_file = split(OPENMPI_URL)[1]
mpi_archive_path = join(options.depdir, mpi_archive_file)
mpi_src_dir = join(options.depdir, (splitext(splitext(mpi_archive_file)[0])[0]))
if not os.path.isfile(mpi_archive_path):
    print "downloading openmpi"
    urlretrieve(OPENMPI_URL, mpi_archive_path, progress)
tarobj = tarfile.open(mpi_archive_path, 'r:gz')
tarobj.extractall(options.depdir)
print "building openmpi"
os.chdir(mpi_src_dir)
if os.system("./configure --prefix=%s" % join(options.prefix)) != 0:
    print "openmpi configuration failed"
    sys.exit()
if os.system("make -j%i" % ncpus()) != 0:
    print "build in %s failed" % open_src_dir
    sys.exit()
if os.system("make install") != 0:
    print "make install in %s failed" % open_src_dir
    sys.exit()
os.chdir(topdir)
print "build of openmpi completed"

###########################
# ITK
###########################
itk_archive_file = split(ITK_URL)[1]
itk_archive_path = join(options.depdir, itk_archive_file)
itk_src_dir = join("..", (splitext(splitext(itk_archive_file)[0])[0]))
itk_build_dir = join(options.depdir, "itk-build")
if not os.path.isfile(itk_archive_path):
    print "Downloading ITK"
    urlretrieve(CMAKE_URL, itk_archive_path, progress)
tarobj = tarfile.open(itk_archive_path, 'r:gz')
tarobj.extractall(options.depdir)
print "Building ITK"
try:
    os.makedirs(itk_build_dir)
except os.error:
    print "Directory %s exists, using it" %itk_build_dir

os.chdir(itk_build_dir)
#if os.system("cmake %s -DCMAKE_INSTALL_PREFIX=%s" % (itk_src_dir, options.prefix)) != 0:
#	print "itk configuration in %s failed" % itk_build_dir
#	sys.exit()
#if os.system("make -j%i" % ncpus()) != 0:
#	print "build in %s failed" % itk_build_dir
#	sys.exit()
#if os.system("make install") != 0:
#	print "build in %s failed" % itk_build_dir
#	sys.exit()
#os.chdir(topdir)
print "Build of ITK Completed"

###########################
# dysii
###########################
dysii_archive_file = split(DYSII_URL)[1]
dysii_archive_path = join(options.depdir, dysii_archive_file)
dysii_src_dir = join(options.depdir, (splitext(splitext(dysii_archive_file)[0])[0]))
dysii_build_dir = join(options.depdir, "dysii-build")
if not os.path.isfile(boost_archive_path):
    print "Downloading dysii-1.4"
    urlretrieve(DYSII_URL, dysii_archive_path, progress)
#tarobj = tarfile.open(dysii_archive_path, 'r:gz')
#tarobj.extractall(options.depdir)
#print "Patching dysii for CMAKE"
#print "Building dysii-1.4"
#os.makedirs(dysii_build_dir)
#os.chdir(dysii_build_dir)
#if os.system("cmake %s" % dysii_src_dir) != 0:
#	print "dysii configuration in %s failed" % dysii_build_dir
#	sys.exit()
#if os.system("make -j%i" % ncpus()) != 0:
#	print "build in %s failed" % dysii_build_dir
#	sys.exit()
#os.chdir(topdir)
#print "Build of dysii Completed"
