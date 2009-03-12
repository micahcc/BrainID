#!env python
import os
from os.path import join, split, splitext 
import sys
from urllib import urlretrieve
import tarfile
from optparse import OptionParser

# some user modifiable constants
CMAKEURL="http://www.cmake.org/files/v2.6/cmake-2.6.2.tar.gz"

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


