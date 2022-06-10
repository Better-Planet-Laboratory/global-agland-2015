### ModuleNotFoundError: No module named '_gdal_array'
Setting up a correct environment with GDAL is not an easy task, and when you build the [Docker Image](../../../Dockerfile) to run the program under Unbuntu, it is possible to encounter ``` ModuleNotFoundError: No module named '_gdal_array' ``` after runnning ``` python deploy.py ```. There are many fixes online, but I found the following to be the only way that worked. 

#### 1. Test If You Have The Issue
After adding the program under a docker container built from the [Docker Image](../../../Dockerfile) (you should see something like ```root@7702bddeadad:``` in terminal), run:
```
python 
>>> from osgeo import gdal
>>> from osgeo import gdal_array
>>> exit()
```
If you don't see anything returned, you can skip the rest of the instruction. No fix needed. Otherwise, go to Section 2.

#### 2. Uninstall numpy and GDAL
To uninstall numpy and GDAL, run:
```
pip uninstall numpy 
pip uninstall GDAL
```

