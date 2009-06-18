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
#ITK_URL="http://voxel.dl.sourceforge.net/sourceforge/itk/InsightToolkit-3.12.0.tar.gz"
ITK_URL="http://voxel.dl.sourceforge.net/sourceforge/itk/InsightToolkit-3.14.0.tar.gz"
#OPENMPI_URL="http://www.open-mpi.org/software/ompi/v1.3/downloads/openmpi-1.3.1.tar.gz"
OPENMPI_URL="http://www.open-mpi.org/software/ompi/v1.2/downloads/openmpi-1.2.9.tar.gz"
LAPACK_URL="http://www.netlib.org/lapack/lapack-3.2.1.tgz"
MATLAB_NIFTI="http://www.pc.rhul.ac.uk/staff/J.Larsson/software/cbiNifti/cbiNifti.tar.gz"


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
        try:
            tarobj.extractall(basedir)
        except AttributeError:
            #for backward compatability
            for tarinfo in tarobj:
                print tarinfo.name
                tarobj.extract(tarinfo, basedir);
   
    if os.path.exists("%s.patch" %name) and os.system("patch --dry-run -Np2 -d %s < %s" %(src_dir, "%s.patch" % name)) == 0:
        print "Patching %s" % name
        os.system("patch -Np2 -d %s < %s" % (src_dir, "%s.patch" % name))
    return src_dir

#defstrings should be a tuple of string arguments to pass to configure
def buildboost(basedir, instdir, name, url):
    topdir = os.getcwd()
    src_dir = getep(basedir, name, url)
#    install_dir = join(instdir, name)
    
    print "building %s" % name
    os.chdir(src_dir)
    if  os.system("./configure --prefix=%s %s" % (instdir, "--with-libraries=serialization,mpi,program_options")) != 0:
        print "%s configuration failed" % name
        sys.exit()
    os.system("echo using mpi \; >> %s/user-config.jam" % src_dir)
    os.system("echo BJAM_CONFIG=\"--layout=system\" >> %s/Makefile" % src_dir)
    if os.system("make -j%i" % ncpus()) != 0:
        print "build in %s failed" % src_dir
        sys.exit()
    
    if os.system("make install") != 0:
        print "make install in %s failed" % src_dir
        sys.exit()
    print "build of %s completed" % name
    
    os.chdir(join(instdir, "lib"));
    liblist = [];
    for file in os.listdir("./"):
        try:
            os.readlink(file)
        except os.error:
            liblist.append(file)

    for file in liblist:
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
    return instdir

#defstrings should be a tuple of string arguments to pass to configure
def confmakeinst(basedir, instdir, name, url, defstrings = ""):
    topdir = os.getcwd()
    src_dir = getep(basedir, name, url)
#    instdir = join(instdir, name)
    
    print "building %s" % name
    os.chdir(src_dir)
    if os.system("make -j%i" % ncpus()) != 0:
        if  os.system("./configure --prefix=%s %s" % (instdir, " ".join(defstrings))) != 0:
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
    return instdir

#defstrings should be a tuple of extra arguments to give to cmake
def cmakeinst(basedir, instdir, name, url, defstrings = ""):
    topdir = os.getcwd()
    src_dir = getep(basedir,name,url)
#    install_dir = join(instdir, name)
    
    print "Building %s" % name
    build_dir = join(basedir, "%s-build" % name)
    try:
        os.makedirs(build_dir)
    except os.error:
        print "Directory %s exists, using it" % build_dir
    
    os.chdir(build_dir)
    if os.system("cmake %s -DCMAKE_INSTALL_PREFIX=%s %s" % (src_dir, instdir, " ".join(defstrings))) != 0:
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
    return instdir

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
                  default="Dependencies-Build",
                  help="name of directory for dependencies")
parser.add_option("-u", "--update",
                  action="store_true", dest="update", default=False,
                  help="run svn update first")
parser.add_option("-i", "--depinstall",
                  action="store_true", dest="depprefix", default="%s/Dependencies-Install" % topdir,
                  help="The location to put the final <depname>/include, <depname>/lib, <depname>/bin ")
parser.add_option("-p", "--prefix",
                  action="store_true", dest="prefix", default="%s" % join(srcpath, "install"),
                  help="The location to put the final include, lib, bin....dirs")
(options, args) = parser.parse_args()

depdir = join(topdir, options.depdir)
depprefix = join(topdir, options.depprefix)

PROFILE_OUT="%s/bash_profile" % topdir
prof_bin = [] 
prof_ld = []


platform = os.uname()[0]
if platform == 'Darwin':
        APPLE_ARCH = os.uname()[4]
        print "Detected platform Darwin, enabling universal binaries"
        CMAKE_APPLE_UNIV = "-D CMAKE_OSX_ARCHITECTURES=i386;ppc -D CMAKE_TRY_COMPILE_OSX_ARCHITECTURES=" + APPLE_ARCH


if os.path.exists(depprefix):
    print "Directory " + depprefix + " exists, removing in..."
    print "3"
    time.sleep(1);
    print "2"
    time.sleep(1);
    print "1"
    time.sleep(1);
    shutil.rmtree(depprefix)

# make directories if they do not exist
try:
    os.makedirs(depdir)
except os.error:
    print "Directory " + depdir + " may overwrite."


##########################
# CMake
##########################
cmake_install_dir = cmakeinst(depdir, depprefix, "cmake", CMAKE_URL)
os.environ["PATH"] = join(cmake_install_dir, "bin") + ":"+ os.environ["PATH"]
if not join(cmake_install_dir, "bin") in prof_bin:
    prof_bin += [join(cmake_install_dir, "bin")];

###########################
# LAPACK
###########################
lapack_src_dir = getep(depdir, "lapack", LAPACK_URL)
lapack_install_dir = join(depprefix, "lib")
print "Creating %s" % lapack_install_dir
try:
    os.makedirs(lapack_install_dir)
except os.error:
    print "Directory " + options.depdir + " may overwrite."

os.chdir(lapack_src_dir)
os.system("make -j%i blaslib" % ncpus())
os.system("make -j%i all" % ncpus())
for filen in os.listdir("./"):
    if splitext(filen)[1] == ".a":
        shared = splitext(filen)[0]+ ".so"
        
        print "Creating %s" % shared
        os.system("ar -x %s; gcc -shared *.o -o %s; rm *.o" % (filen, shared))
        
        print "Installing %s" % shared
        shutil.copy(shared, lapack_install_dir)

        print "Installing %s" % shared + ".0"
        os.symlink(join(lapack_install_dir, shared), join(lapack_install_dir, shared+".0"))
        
        print "Installing %s" % shared + ".0.0.0"
        os.symlink(join(lapack_install_dir, shared), join(lapack_install_dir, shared+".0.0.0"))
        
        print "Installing %s" % filen
        shutil.copy(filen, lapack_install_dir)

if not lapack_install_dir in prof_ld:
    prof_ld += [lapack_install_dir];
print prof_ld, prof_bin
#        os.
#os.path.
os.chdir(topdir)
exit
###########################
# gsl
###########################
gsl_install_dir = confmakeinst(depdir, depprefix, "gsl", GSL_URL)
os.environ["PATH"] = join(gsl_install_dir, "bin") + ":"+ os.environ["PATH"]
if not join(gsl_install_dir, "bin") in prof_bin:
    prof_bin += [join(gsl_install_dir, "bin")];
if not join(gsl_install_dir, "lib") in prof_ld:
    prof_ld += [join(gsl_install_dir, "lib")];
print prof_ld, prof_bin

###########################
# openmpi
###########################
mpi_install_dir = confmakeinst(depdir, depprefix, "mpi", OPENMPI_URL)
os.environ["PATH"] = join(mpi_install_dir, "bin") + ":"+ os.environ["PATH"]
if not join(mpi_install_dir, "bin") in prof_bin:
    prof_bin += [join(mpi_install_dir, "bin")];
if not join(mpi_install_dir, "lib") in prof_ld:
    prof_ld += [join(mpi_install_dir, "lib")];
print prof_ld, prof_bin

############################
# Boost 
############################
boost_install_dir = buildboost(depdir, depprefix, "boost", BOOST_URL);
if not join(boost_install_dir, "lib") in prof_ld:
    prof_ld += [join(boost_install_dir, "lib")];
print prof_ld, prof_bin

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
try:
    tarobj.extractall(depdir)
except AttributeError:
    #for backward compatability
    for tarinfo in tarobj:
        print tarinfo.name
        tarobj.extract(tarinfo, depdir);

shutil.move(join(depdir, "boost-numeric-bindings", "boost", "numeric", "bindings"), \
            join(boost_install_dir, "include", "boost", "numeric","bindings"))
os.chdir(topdir)

###########################
# ITK
###########################
itk_install_dir = cmakeinst(depdir, depprefix, "itk", ITK_URL, ("-DBUILD_TESTING=OFF", "-DBUILD_EXAMPLES=OFF"))
os.environ["PATH"] = join(itk_install_dir, "bin") + ":"+ os.environ["PATH"]
if not join(itk_install_dir, "bin") in prof_bin:
    prof_bin += [join(itk_install_dir, "bin")];
if not join(itk_install_dir, "lib") in prof_ld:
    prof_ld += [join(itk_install_dir, "lib")];
print prof_ld, prof_bin

###########################
# dysii
###########################
dysii_install_dir = cmakeinst(depdir, depprefix, "dysii", DYSII_URL, ("-DGSL=%s" % gsl_install_dir, \
            "-DMPI=%s" % mpi_install_dir, "-DBOOST=%s" % boost_install_dir))
if not join(dysii_install_dir, "lib") in prof_ld:
    prof_ld += [join(dysii_install_dir, "lib")];
print prof_ld, prof_bin

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

#try:
#    os.makedirs(brainid_install_dir)
#except os.error:
#    print "Directory %s exists, using it" %brainid_install_dir

os.chdir(brainid_build_dir)
if os.system("cmake %s -DITK_DIR=%s " % (srcpath, join(itk_install_dir, "lib", "InsightToolkit")) \
                + " -Ddysii_INCLUDE_DIRS=%s " % join(dysii_install_dir, "include")\
                + " -Ddysii_LIBRARY_DIRS=%s " % join(dysii_install_dir, "lib")\
                + " -DBOOST_INCLUDE_DIRS=%s " % join(boost_install_dir, "include")\
                + " -DBOOST_LIBRARY_DIRS=%s " % join(boost_install_dir, "lib")\
                + " -DMPI_INCLUDE_DIRS=%s " % join(mpi_install_dir, "include")\
                + " -DMPI_LIBRARY_DIRS=%s " % join(mpi_install_dir, "lib")\
                + " -DGSL_INCLUDE_DIRS=%s " % join(gsl_install_dir, "include")\
                + " -DGSL_LIBRARY_DIRS=%s " % join(gsl_install_dir, "lib")\
                + " -DCMAKE_INSTALL_PREFIX=%s " % brainid_install_dir) != 0:
    print "brainid configuration in %s failed" % brainid_build_dir
    sys.exit()

if os.system("make -j%i" % ncpus()) != 0:
    print "build in %s failed" % brainid_build_dir
    sys.exit()

#if os.system("make install") != 0:
#    print "install from %s failed" % brainid_build_dir
#    sys.exit()

#if not join(brainid_install_dir, "bin:") in prof_bin:
#    prof_bin += join(brainid_install_dir, "bin:");
prof_bin += [join(brainid_build_dir, "code")]

os.chdir(topdir)
print "Build of brainid Completed"
print "Copying scripts to %s/%s" % (brainid_build_dir, "code")
shutil.copy("plotout.py", join(brainid_build_dir, "code"))
shutil.copy("simall.py", join(brainid_build_dir, "code"))
shutil.copy("plot.py", join(brainid_build_dir, "code"))


print "Writing out bash script"

FILE = open(PROFILE_OUT, "w")
FILE.write("#!/bin/bash\n")
for ldd in prof_ld:
    FILE.write("LD_LIBRARY_PATH=\"" + ldd + ":$LD_LIBRARY_PATH\"\n")
FILE.write("export LD_LIBRARY_PATH\n")
for bin in prof_bin:
    FILE.write("PATH=\"" + bin + ":$PATH\"\n")
FILE.close()
