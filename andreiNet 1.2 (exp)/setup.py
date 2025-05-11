import os
import sys
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools._distutils import ccompiler
import pybind11 # Import pybind11

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if self.compiler is None:
            self.compiler = ccompiler.new_compiler(
                compiler=getattr(self, 'compiler_name', None),
                verbose=self.verbose,
                dry_run=self.dry_run,
                force=self.force
            )

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        # Get pybind11's CMake directory
        pybind11_cmake_dir = pybind11.get_cmake_dir()

        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable,
                      '-Dpybind11_DIR=' + pybind11_cmake_dir] # Pass pybind11_DIR to CMake

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
        
        if self.compiler.compiler_type == "msvc":
            cmake_args += [f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"]
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
            cmake_args += [f"-DPYBIND11_PYTHON_VERSION={python_version}"]
        else: 
            if os.name != 'nt':
                build_args += ['--', '-j2']


        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        print(f"CMake build temp: {self.build_temp}")
        print(f"CMake source dir: {ext.sourcedir}")
        print(f"CMake args: {cmake_args}")
        
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        
        print(f"CMake build args: {build_args}")
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

setup(
    name='andreinet_py_gui',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='Python GUI for AndreiNet C++ library',
    long_description='',
    ext_modules=[CMakeExtension('andreinet_py', sourcedir='bindings')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "customtkinter",
        "matplotlib",
        "pandas",
        "numpy" 
    ],
    # pybind11 is a build-time dependency, ensure it's installed in your environment
    # setup_requires=['pybind11>=2.6'], # Can also add here for build-time check
)