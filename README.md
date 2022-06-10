# PythonMeshManipulation
## Using a monocular camera to get depth predictions

Using pydepth to estimate depth from a monocular camera. 
This is done as opposed to stereo to improve generalisation of mapping between different platforms. 
Stereo's farfield resolution is directly coupled to the baseline as well. For an algorithm to be useful on smaller robots and drones, using a large baseline is not feasable.



aru_core_lib is a symlink pointing to the location of the python bindings from aru_core
