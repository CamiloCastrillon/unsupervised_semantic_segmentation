import subprocess
import os
import open3d as o3d

def initialize_paths():
    project_dir         = 'C:/camilo/point_cloud/'
    images_dir          = os.path.join(project_dir, 'images/')
    matches_dir         = os.path.join(project_dir, 'matches/')
    reconstruction_dir  = os.path.join(project_dir, 'reconstruction_sequential/')

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(matches_dir, exist_ok=True)
    os.makedirs(reconstruction_dir, exist_ok=True)

    return images_dir, matches_dir, reconstruction_dir

def run_command(command):
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f'Error: {result.stderr}')
    else:
        print(result.stdout)
    
    return result

def plot_cloud_point(reconstruction_dir):
    # Visualizar la nube de puntos con open3d
    pcd = o3d.io.read_point_cloud(f'{reconstruction_dir}scene_dense.ply')
    o3d.visualization.draw_geometries([pcd])
    
def work_flow():
    print('\n🔽 Ejecutando el flujo de trabajo:')
    
    # Inizializa las rutas del proyecto
    print('   🟢 Inizializando las rutas del proyecto.')
    images_dir, matches_dir, reconstruction_dir = initialize_paths()
    
    # Inizializa la lista de imagenes
    print('   🟢 Inizializando la lista de imagenes.')
    command     = f'openMVG_main_SfMInit_ImageListing -i {images_dir} -o {matches_dir} -c 1 -f 37200'
    #run_command(command)
    
    # Obtiene las caracteristicas
    print('   🟢 Obteniendo características.')
    command = (f'openMVG_main_ComputeFeatures -i {matches_dir}sfm_data.json -o {matches_dir}')
    #run_command(command)
    
    # Computa las coincidencias
    print('   🟢 Observando coincidencias.')
    command = (f'openMVG_main_ComputeMatches -i {matches_dir}sfm_data.json -o {matches_dir}matches.txt -f 1')
    #run_command(command)
    
    # Reconstruye caracteristicas
    print('   🟢 Reconstruyendo caracteristicas.')
    command = (f'openMVG_main_SfM -i {matches_dir}sfm_data.json --match_file {matches_dir}matches.txt -o {reconstruction_dir}')
    #run_command(command)
    
    # Convierte a formato mvs
    print('   🟢 Convirtiendo de formato mvg a mvs.')
    command = (f'openMVG_main_openMVG2openMVS -i {reconstruction_dir}sfm_data.bin -o {reconstruction_dir}scene.mvs')
    #run_command(command)
    
    # Densifica la nube de puntos
    print('   🟢 Densificando nube de puntos.')
    command = (f'DensifyPointCloud {reconstruction_dir}scene.mvs')
    #run_command(command)
    
    # Convierte la nube de puntos a un formato diferente
    print('   🟢 Convirtiendo de formato mvs a ply.')
    command = (f'ReconstructMesh -i {reconstruction_dir}scene_dense.mvs -o {reconstruction_dir}scene_dense.ply')
    run_command(command)
    
    # Grafica la nube de puntos
    print('   🟢 Graficando la nube de puntos.')
    plot_cloud_point(reconstruction_dir)

def main():
    try:
        work_flow()
    except Exception as e:
        print(f'\n⛔ Error:\n{e}\n')

# Ejecuta el flujo de trabajo
main()
