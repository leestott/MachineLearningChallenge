# Machine Learning Challenge 

## Data Set

The data set is available at https://challenge.blob.core.windows.net/challengefiles/gear_images.zip at 33mb

To Download simply download via the url or run the following command on the DSVM 

curl -O https://challenge.blob.core.windows.net/challengefiles/gear_images.zip
unzip *.zip

Or in windows powershell;

wget "https://challenge.blob.core.windows.net/challengefiles/gear_images.zip" -OutFile gear_images.zip
unzip gear_images.zip

## Running Notebooks 

Running Notebooks on Azure Notebook Service see [Here](http://notebooks.azure.com) 

This repo is available as Azure Notebooks [Here](https://notebooks.azure.com/LeeStott-Microsoft/projects/machinelearningchallenge)

Running the Notebook on Azure Data Science Virtual Machine 
See [Here](https://blogs.msdn.microsoft.com/uk_faculty_connection/2018/12/10/microsoft-azure-notebooks-and-additional-compute-capacity-via-connecting-to-data-science-vms/)

## Running Notebooks on Windows Desktop 

Preq is having C++ 9 for Python download the Microsoft Visual C++ 9.0 is required. Get it from [http://aka.ms/vcpython2](http://aka.ms/vcpython2)

Make your way over to python.org, download and install the latest version (3.7 as of this writing) and make sure that wherever you install it, the directory containing python.exe is in your system PATH environment variable. I like to install it in the root of my C: drive, e.g. C:\Python37, so my PATH contains that directory.
Once that's installed, you'll want to create a virtual environment, a lightweight, disposable, isolated python installation where you can experiment and install 3rd party libraries without affecting your "main" installation. To do this, open up a Powershell window, and enter the following commands (where "myenv" is the name of the virtualenv we're going to create, you can use any name you like for this):

Open a Powershell windows 

PS C:\> python -m venv myenv

PS C:\> myenv\Scripts\activate

Then, let's install jupyter and start up a notebook:

PS C:\> pip install jupyter

PS C:\> jupyter notebook

Incidentally, if you get a warning about upgrading pip, make sure to use the following incantation to upgrade (to prevent an issue on windows where pip is unable to upgrade its own executable in-place):
PS C:\> python -m pip install --upgrade pip

Advantages: Uses "pure" python, official tools, and no external dependencies. Well supported, with plenty of online documentation and support communities.

Disadvantages: While many popular data analysis or scientific python libraries can be installed by pip on windows (including Pandas and Matplotlib), some (for example SciPy) require a C compiler and the presence of 3rd party C libraries on the system which are difficult to install on Windows.

## Working with Jupyter Notebooks in Visual Studio Code

See [https://code.visualstudio.com/docs/python/jupyter-support](https://code.visualstudio.com/docs/python/jupyter-support)