'''
Created on Mar 8, 2022

@author: ricardo
'''
import numpy as np
from stl import mesh
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import pandas as pd
    
    
class BoatMesh(object):
    def __init__(self, fname : str = None, reference = [[0,0,0]]):
        self.mesh       = mesh.Mesh.from_file(fname)
        mass_prop       = self.mesh.get_mass_properties()
        self.volume     = mass_prop[0]
        self.cg         = mass_prop[1].reshape(1,3)
        self.cob        = None
        self.mc         = None
        self.reference  = np.array(reference)
        self.phi        = 0
        self.theta      = 0
        self.waterline_offset   = 0
        self.mesh_update()
        
    def set_attitude(self, phi = 0, theta = 0):
        self.reset_attitude()
        self.mesh.rotate([1,0,0], phi, self.cg[0])
        self.mesh.rotate([0,1,0], theta, self.cg[0])
        self.phi        = phi
        self.theta      = theta
        self.mesh_update()
        
    def reset_attitude(self):
        self.mesh.rotate([0,1,0], -self.theta, self.cg[0])
        self.mesh.rotate([1,0,0], -self.phi, self.cg[0])
        self.phi        = 0
        self.theta      = 0
        self.mesh_update()
    
    def print_mesh_info(self):
        print("Total volume: {:.3f}".format(self.volume))
        print("Bounding box: ")
        print("X : {:.3f}, {:.3f}".format(np.max(self.mesh.x), np.min(self.mesh.x)))
        print("Y : {:.3f}, {:.3f}".format(np.max(self.mesh.y), np.min(self.mesh.y)))
        print("Z : {:.3f}, {:.3f}".format(np.max(self.mesh.z), np.min(self.mesh.z)))
        print("Total triangles: {:d}".format(self.mesh.points.shape[0]))
        
    def mesh_update(self):
        self.mesh.update_normals()
        self.mesh.update_units()
        self.refresh_centroids() 
        
    def refresh_centroids(self):
        centroids_x = (self.mesh.points[:,0] + self.mesh.points[:,3] + self.mesh.points[:,6])/3
        centroids_y = (self.mesh.points[:,1] + self.mesh.points[:,4] + self.mesh.points[:,7])/3
        centroids_z = (self.mesh.points[:,2] + self.mesh.points[:,5] + self.mesh.points[:,8])/3

        self.centroids = np.transpose(np.vstack([centroids_x, centroids_y, centroids_z]))
        
    def display_mesh(self):
        # Create a new plot
        figure = plt.figure()
        axes = mplot3d.Axes3D(figure)
        
        # Load the STL files and add the vectors to the plot
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(self.mesh.vectors))
        
        # Auto sale to the mesh size
        scale = self.mesh.points.flatten()
        axes.auto_scale_xyz(scale, scale, scale)
        
        # Show the plot to the screen
        plt.show()
        
    def calc_buoyancy(self, rho : float = 1000, g : float = 9.81, verbose = False):
        pos_vec     = self.reference - m.centroids
                                 
        dF          = rho * g * (
                            np.transpose([self.centroids[:,2]]) + self.waterline_offset
                            )* self.mesh.areas
        dF[dF < 0]  = 0
        dF          = dF * self.mesh.units
         
        dM = np.cross(dF, pos_vec)
        
        F = np.sum(dF,0)
        M = np.sum(dM,0) 
        
        Fvec = np.array([[    0, -F[2], F[1]],
                         [ F[2],     0,-F[0]],
                         [-F[1],  F[0],   0]])
        B = self.reference + np.dot(np.reshape(M,[1,3]),np.linalg.pinv(Fvec))
        
        if self.phi**2 > 1e-2:
            t = (B[0,1] - (np.tan(self.phi) - self.reference[0,2])*B[0,2]) / (F[2]*np.tan(self.phi) - F[1])
            MC = B + t * F
        else:
            MC = np.array([[np.nan, np.nan, np.nan]])
            
        self.mc = MC
        
        if verbose:
            print("*** Buoyancy calculation subroutine ***")
            print("Ref (x,y,z): {:.3f}, {:.3f}, {:.3f}".format(*self.reference[0]))
            print("CoB (x,y,z): {:.3f}, {:.3f}, {:.3f}".format(*B[0]))
            print("MCt (x,y,z): {:.3f}, {:.3f}, {:.3f}".format(*self.mc[0]))
            print("F @ ref: {:.1f}, {:.1f}, {:.1f}".format(*F))
            print("M @ ref: {:.1f}, {:.1f}, {:.1f}".format(*M))
  
        return F, M, B, MC
    
    def floating_height(self, Fz_target, rho : float = 1000, g : float = 9.81, verbose = False):
        max_buoyancy = rho * g * self.volume
        if Fz_target >= max_buoyancy or Fz_target <= 0:
            raise(ValueError(
                "Expected buoyant force {:.1f} exceeds bounds (max {:.1f}, min 0)".format(Fz_target, max_buoyancy))
            )
            
        Fz = 0
        self.waterline_offset = 0
        i = 1
        
        while (Fz_target - Fz)**2 > 1e-2:
            F, _, _, _ = self.calc_buoyancy(rho, g, verbose)
            Fz = F[2]
            dF = Fz - Fz_target
            self.waterline_offset += -(dF/(self.volume*rho*g))*0.01
            i += 1
            if verbose:
                print("Current waterline height: {:.3}, iter {:d}".format(self.waterline_offset, i))
            
        return self.waterline_offset
    

"""
Simulation parameters
"""
if __name__ == "__main__":
    rho     = 1000
    g       = 9.81
    
    m = BoatMesh("bo3.stl")
    m.print_mesh_info()
    m.display_mesh()
    
    _, _, B, _ = m.calc_buoyancy(rho, g, True)

    m.reference = B
    m.calc_buoyancy(rho, g, True)
    
    m.floating_height(Fz_target = 880, 
                      rho = rho, 
                      g = g, 
                      verbose = True)
    
    #print("Center of buoyancy: {:.2f}, {:.2f}, {:.2f}".format(*ref[0]))
    #print("F @ COB: {:.2f}, {:.2f}, {:.2f}".format(*F))
    #print("M @ COB: {:.2f}, {:.2f}, {:.2f}".format(*M))
    
    data = []
    
    phi_list = np.linspace(-10,10,50)*np.pi/180
    
    
    Fz_target = 72*9.81
    
    for phi in phi_list:
        m.set_attitude(phi = phi, theta = 0)
        m.floating_height(Fz_target)
        F, M, B, MC = m.calc_buoyancy(verbose = True)
        
        data.append({
            "phi" : phi,
            "theta" : 0,
            "F" : F,
            "M" : M,
            "B" : B,
            "MC" : MC})
        
    df = pd.DataFrame(data)
    fig = plt.figure()
    
    plt.subplot(2,3,1)
    plt.plot(df["phi"]*180/np.pi, [x[0] for x in df["M"]])
    plt.title("Rolling moment")
    plt.xlabel(r"$\phi$ (deg)")
    plt.ylabel("Righting moment (Nm)")
    plt.grid()
    
    plt.subplot(2,3,2)
    plt.plot([x[0][0] for x in df["B"]], [x[0][2] for x in df["B"]])
    plt.title("CoB Position")
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.grid()
    
    plt.subplot(2,3,3)
    plt.plot([x[0][0] for x in df["MC"]], [x[0][2] for x in df["MC"]])
    plt.title("Metacenter Position")
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.grid()
        
    theta_list = np.linspace(-10,10,50)*np.pi/180
    data = []
    for theta in theta_list:
        m.set_attitude(phi = 0, theta = theta)
        m.floating_height(Fz_target)
        F, M, B, MC = m.calc_buoyancy(verbose = True)
        
        data.append({
            "phi" : 0,
            "theta" : theta,
            "F" : F,
            "M" : M,
            "B" : B,
            "MC" : MC})
        
    df = pd.DataFrame(data)

    
    plt.subplot(2,3,4)
    plt.plot(df["theta"]*180/np.pi, [x[1] for x in df["M"]])
    plt.title("Piching moment")
    plt.xlabel(r"$\theta$ (deg)")
    plt.ylabel("Piching moment (Nm)")
    plt.grid()
    
    plt.subplot(2,3,5)
    plt.plot([x[0][0] for x in df["B"]], [x[0][2] for x in df["B"]])
    plt.title("CoB Position")
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.grid()
    
    plt.subplot(2,3,6)
    plt.plot([x[0][0] for x in df["MC"]], [x[0][2] for x in df["MC"]])
    plt.title("Metacenter Position")
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.grid()
    
    plt.show()
