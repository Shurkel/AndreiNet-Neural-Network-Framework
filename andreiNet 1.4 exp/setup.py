# setup.py
import sys
from setuptools import setup, Extension
# Ensure pybind11 is available for setup helpers
try:
    import pybind11
except ImportError:
    print("Error: pybind11 is required to build this package.")
    print("Please run: pip install pybind11")
    sys.exit(1)

# Define compiler/linker args (example for g++)
# Adjusted for potential MSVC on Windows
cpp_args = []
link_args = []
if sys.platform == 'win32':
    cpp_args = ['/std:c++11', '/O2', '/EHsc'] # MSVC flags
else: # Assume Unix-like (Linux/macOS)
    cpp_args = ['-std=c++11', '-O2']
    if sys.platform == 'darwin': # macOS specific flags
        cpp_args.extend(['-stdlib=libc++', '-mmacosx-version-min=10.9'])
        link_args.extend(['-stdlib=libc++', '-mmacosx-version-min=10.9'])
    # Enable OpenMP if desired (check compiler support)
    # cpp_args.append('-fopenmp')
    # link_args.append('-fopenmp')


# Define the extension module
ext_modules = [
    Extension(
        # Module name (must match PYBIND11_MODULE name in bindings.cpp)
        'andreinet_bindings',
        # Source files
        ['bindings/bindings.cpp'],
        # Include directories
        include_dirs=[
            pybind11.get_include(),
            'cpp_src', # Directory containing your andreiNET headers
            'eigen'    # Directory containing the Eigen library headers (e.g., 'eigen/Eigen')
        ],
        # Language specification
        language='c++',
        # Compiler arguments
        extra_compile_args=cpp_args,
        # Linker arguments
        extra_link_args=link_args,
    ),
]

setup(
    name='andreinet_bindings',
    version='1.3.0', # Match your library version
    author='Your Name', # Replace with your name
    author_email='your_email@example.com', # Replace with your email
    description='Python bindings for andreiNET',
    long_description='', # Optional long description
    ext_modules=ext_modules,
    # --- NO cmdclass line ---
    zip_safe=False, # Recommended for C extensions
    python_requires='>=3.6', # Specify minimum Python version
)