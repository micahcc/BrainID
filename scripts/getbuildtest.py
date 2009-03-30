#!env python
import os
from os.path import join, split, splitext 
import sys
from urllib import urlretrieve
import tarfile
from optparse import OptionParser

# some user modifiable constants
CMAKEURL="http://www.cmake.org/files/v2.6/cmake-2.6.2.tar.gz"
DYSIIURL="http://www.indii.org/files/dysii/releases/dysii-1.4.0.tar.gz"
GSLURL="ftp://ftp.gnu.org/gnu/gsl/gsl-1.9.tar.gz"
BOOSTURL="http://downloads.sourceforge.net/boost/boost_1_38_0.tar.gz?use_mirror=voxel"

BINDINGSGIT="http://git.tiker.net/trees/boost-numeric-bindings.git"

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

(options, args) = parser.parse_args()

platform = os.uname()[0]
if platform == 'Darwin':
	APPLE_ARCH = os.uname()[4]
	print "Detected platform Darwin, enabling universal binaries"
	CMAKE_APPLE_UNIV = "-D CMAKE_OSX_ARCHITECTURES=i386;ppc -D CMAKE_TRY_COMPILE_OSX_ARCHITECTURES=" + APPLE_ARCH


# make directories if they do not exist
try:
	os.makedirs(options.depdir)
except os.error:
	print "Directory " + options.depdir + " exists. Halting."
	sys.exit()


##########################
# CMake
##########################
print "Downloading CMake"
cmake_archive_file = split(CMAKEURL)[1]
cmake_archive_path = join(options.depdir, cmake_archive_file)
cmake_src_dir = join("..", (splitext(splitext(cmake_archive_file)[0])[0]))
cmake_build_dir = join(options.depdir, "cmake-build")
urlretrieve(CMAKEURL, cmake_archive_path, progress)
tarobj = tarfile.open(cmake_archive_path, 'r:gz')
tarobj.extractall(options.depdir)
print "Building CMake"
os.makedirs(cmake_build_dir)
os.chdir(cmake_build_dir)
if os.system("cmake %s" % cmake_src_dir) != 0:
	print "cmake configuration in %s failed" % cmake_build_dir
	sys.exit()
if os.system("make -j%i" % ncpus()) != 0:
	print "build in %s failed" % cmake_build_dir
	sys.exit()
os.chdir(topdir)
print "Build of CMake Completed"

###########################
# gsl
###########################
print "Downloading gsl"
gsl_archive_file = split(GSLURL)[1]
gsl_archive_path = join(options.depdir, gsl_archive_file)
gsl_src_dir = join("..", (splitext(splitext(gsl_archive_file)[0])[0]))
gsl_build_dir = join(options.depdir, "gsl-build")
urlretrieve(GSLURL, gsl_archive_path, progress)
tarobj = tarfile.open(gsl_archive_path, 'r:gz')
tarobj.extractall(options.depdir)
print "Building gsl"
os.makedirs(gsl_build_dir)
os.chdir(gsl_build_dir)
if os.system("cmake %s" % gsl_src_dir) != 0:
	print "gsl configuration in %s failed" % gsl_build_dir
	sys.exit()
if os.system("make -j%i" % ncpus()) != 0:
	print "build in %s failed" % gsl_build_dir
	sys.exit()
os.chdir(topdir)
print "Build of gsl Completed"

############################
# Boost 
############################
#build serialization


############################
# Boost Numeric Bindings
############################
print "Boost Numeric Bindings"
boost-numeric-bindings_archive_file = split(BINDINGSURL)[1]
boost-numeric-bindings_archive_path = join(options.depdir, boost-numeric-bindings_archive_file)
boost-numeric-bindings_src_dir = join("..", (splitext(splitext(boost-numeric-bindings_archive_file)[0])[0]))
boost-numeric-bindings_build_dir = join(options.depdir, "boost-numeric-bindings-build")
urlretrieve(BINDINGSURL, boost-numeric-bindings_archive_path, progress)
tarobj = tarfile.open(boost-numeric-bindings_archive_path, 'r:gz')
tarobj.extractall(options.depdir)
os.chdir(topdir)
print "Build of boost-numeric-bindings Completed"

###########################
# dysii
###########################
print "Downloading dysii-1.4"
dysii_archive_file = split(DYSIIURL)[1]
dysii_archive_path = join(options.depdir, dysii_archive_file)
dysii_src_dir = join("..", (splitext(splitext(dysii_archive_file)[0])[0]))
dysii_build_dir = join(options.depdir, "dysii-build")
urlretrieve(DYSIIURL, dysii_archive_path, progress)
tarobj = tarfile.open(dysii_archive_path, 'r:gz')
tarobj.extractall(options.depdir)
print "Patching dysii for CMAKE"
print "Building dysii-1.4"
os.makedirs(dysii_build_dir)
os.chdir(dysii_build_dir)
if os.system("cmake %s" % dysii_src_dir) != 0:
	print "dysii configuration in %s failed" % dysii_build_dir
	sys.exit()
if os.system("make -j%i" % ncpus()) != 0:
	print "build in %s failed" % dysii_build_dir
	sys.exit()
os.chdir(topdir)
print "Build of dysii Completed"
