'''
Created on Mar 8, 2022

@author: ricardo
'''
import numpy as np
import json

from stl import mesh
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import copy

def plot_stl_matplotlib(stl_mesh):
    figure = pyplot.figure()
    axes = mplot3d.Axes3D(figure)
    
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(stl_mesh.vectors))
    
    # Auto scale to the mesh size
    scale = stl_mesh.points.flatten()
    axes.auto_scale_xyz(scale, scale, scale)
    
    # Show the plot to the screen
    pyplot.show()

def calc_volume_from_stl(envelope, x = None, theta = None, faces = None) :
    stl_mesh, _ = gen_mesh(envelope, x, theta, faces)
    volume, _, _ = stl_mesh.get_mass_properties()
    return np.abs(volume)


def refresh_centroids(m : mesh):
    centroids_x = (m.points[:,0] + m.points[:,3] + m.points[:,6])/3
    centroids_y = (m.points[:,1] + m.points[:,4] + m.points[:,7])/3
    centroids_z = (m.points[:,2] + m.points[:,5] + m.points[:,8])/3
    
    centroids   = np.transpose(np.vstack([centroids_x, centroids_y, centroids_z]))
    m.centroids = centroids
    return centroids

def calc_buoyancy(m : mesh, ref : np.array, rho : float = 1000, g : float = 9.81):
    pos_vec     = ref - m.centroids
                             
    dF          = rho * g * np.transpose([m.centroids[:,2]]) * m.areas
    dF[dF < 0]  = 0
    dF          = dF * m.units
     
    dM = np.cross(dF, pos_vec)
    
    F = np.sum(dF,0)
    M = np.sum(dM,0) 
    
    return F, M 

def calc_center_of_buoyancy(ref, F, M):
    Fvec = np.array([[    0, -F[2], F[1]],
                     [ F[2],     0,-F[0]],
                     [-F[1],  F[0],   0]])
    B = ref + np.dot(np.reshape(M,[1,3]),np.linalg.pinv(Fvec))
    return B

"""
Simulation parameters
"""
if __name__ == "__main__":
    rho     = 1000
    g       = 9.81
    
    m = mesh.Mesh.from_file("bo2.stl", calculate_normals = True)
    print("Total volume: {:.3f} m**3".format(m.get_mass_properties()[0]))
    print("Bounding box: ")
    print("X : {:.3f}, {:.3f}".format(np.max(m.x), np.min(m.x)))
    print("Y : {:.3f}, {:.3f}".format(np.max(m.y), np.min(m.y)))
    print("Z : {:.3f}, {:.3f}".format(np.max(m.z), np.min(m.z)))
    
    # # Create a new plot
    # figure = plt.figure()
    # axes = mplot3d.Axes3D(figure)
    #
    # # Load the STL files and add the vectors to the plot
    # axes.add_collection3d(mplot3d.art3d.Poly3DCollection(m.vectors))
    #
    # # Auto scale to the mesh size
    # scale = m.points.flatten()
    # axes.auto_scale_xyz(scale, scale, scale)
    #
    # # Show the plot to the screen
    # plt.show()
    
    m.update_normals()
    m.update_units()
    refresh_centroids(m)
    
    ref = np.array([[0,0.000,0]])
    
    F, M = calc_buoyancy(m, ref, rho, g)
    B    = calc_center_of_buoyancy(ref, F, M)
    print("Ref: {:.2f}, {:.2f}, {:.2f}".format(*ref[0]))
    print("F @ ref: {:.2f}, {:.2f}, {:.2f}".format(*F))
    print("M @ ref: {:.2f}, {:.2f}, {:.2f}".format(*M))
    
    #ref     = B
    #F, M    = calc_buoyancy(m, ref, rho, g)
    #B       = calc_center_of_buoyancy(ref, F, M)
    
    #print("Center of buoyancy: {:.2f}, {:.2f}, {:.2f}".format(*ref[0]))
    #print("F @ COB: {:.2f}, {:.2f}, {:.2f}".format(*F))
    #print("M @ COB: {:.2f}, {:.2f}, {:.2f}".format(*M))
    
    M_list = []
    B_list = []
    MC_list = []
    
    phi_list = np.linspace(-30,30)
    m_orig = copy.deepcopy(m)
    Fz_orig = F[2]
    
    for phi in phi_list:
        m = copy.deepcopy(m_orig)
        m.rotate(np.array([1,0,0]), phi*np.pi/180, [0,0,0])
        m.update_normals()
        m.update_units()
        refresh_centroids(m)
        
        Fz = 0
        while (Fz_orig - Fz)**2 > 1e-4:
            F, M = calc_buoyancy(m, ref, rho, g)  
            Fz = F[2]
            dF = Fz - Fz_orig
            m.translate([0,0,-dF*0.00001])
            m.update_normals()
            m.update_units()
            refresh_centroids(m)
        
        B    = calc_center_of_buoyancy(ref, F, M)
        
        Fn = F / np.linalg.norm(F)
        t = - B[0,1] / F[1]
        MC = B + t * Fn
        
        M_list.append(M) 
        B_list.append(B[0])
        MC_list.append(MC[0])
        print(F, M, B)
    
    M_list = np.array(M_list)
    B_list = np.array(B_list)
    MC_list = np.array(MC_list)
    
    plt.subplot(1,3,1)
    plt.plot(phi_list, M_list[:,0])
    plt.title("Rolling moment")
    plt.xlabel(r"$\theta$ (deg)")
    plt.ylabel("Righting moment (Nm)")
    plt.grid()
    
    plt.subplot(1,3,2)
    plt.plot(B_list[:,1], B_list[:,2], "k.")
    plt.title("CoB Position")
    plt.xlabel("y (m)")
    plt.ylabel("z (m)")
    plt.grid()
    
    plt.subplot(1,3,3)
    plt.plot(MC_list[:,1], MC_list[:,2], "k.")
    plt.title("Metacender Position")
    plt.xlabel("y (m)")
    plt.ylabel("z (m)")
    plt.grid()
    
    plt.show()
