#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'notebooks'))
	print(os.getcwd())
except:
	pass

#%%
get_ipython().system(u' pip freeze')
get_ipython().system(u' pip --version')


#%%
import sys
sys.version


#%%



