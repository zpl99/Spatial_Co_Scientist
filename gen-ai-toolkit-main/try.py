import subprocess

env_name = "arcgispro"
script_path = r"C:\Users\64963\OneDrive - The University of Texas at Austin\PhD\6-Job\ESRI\Spatial_Co_Scientist\gen-ai-toolkit-main\zp_toolboxes\PrepareExistingLocations.py"
conda_path = r"D:\MyTool\anaconda\Scripts\conda.exe"
project_path = r"C:\Users\64963\Documents\ArcGIS\Projects\MyProject9\MyProject9.aprx"
cmd = f'"{conda_path}" run -n {env_name} python "{script_path}" --project_path "{project_path}"'
# cmd = f'"{conda_path}" run -n {env_name} python "{script_path}" --project_path "C:/Users/64963/Documents/ArcGIS/Projects/MyProject9/MyProject9.aprx"'

result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

print("STDOUT:\n", result.stdout)
print("STDERR:\n", result.stderr)