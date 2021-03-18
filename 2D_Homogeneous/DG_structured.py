from fenics import *
from mshr import *
from matplotlib import pyplot as plt
import numpy as np
import math,sys,time
# FIXME: Make mesh ghosted
parameters["ghost_mode"] = "shared_facet"
class right(SubDomain):
    def inside(self,x,on_boundary):
        return on_boundary and near(x[0],100.0)

class left(SubDomain):
    def inside(self,x,on_boundary):
        return on_boundary and near(x[0],0.0)
right = right()
left = left()


T = 120            # final time
num_steps = 2     # number of time steps
dt = T / num_steps # time step size
phi= Constant(0.2)    # Porosity
#=====================================;
#  Create mesh and identify boundary  ;
#=====================================;
nx = 2
order = 1
mesh = RectangleMesh(Point(0,0),Point(100,100),nx,nx)
boundaries = MeshFunction("size_t",mesh, mesh.topology().dim()-1)
boundaries.set_all(0)
left.mark(boundaries,1)
right.mark(boundaries,2)

ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
#==========================;
#  Define function spaces ;
#=========================;
sSpace_FE = FiniteElement("DG", mesh.ufl_cell(),order)
pSpace_FE = FiniteElement("DG", mesh.ufl_cell(),order)
sSpace = FunctionSpace(mesh,sSpace_FE)
pSpace = FunctionSpace(mesh,pSpace_FE)
wSpace = FunctionSpace(mesh, sSpace_FE * pSpace_FE)
kSpace = FunctionSpace(mesh,"DG",0)
ErrorSpace = FunctionSpace(mesh,"DG",0)
#===================================;
#  Define trial and test functions  ;
#===================================;
# (s,p) = (u[0],u[1]) # for non-linear solvers
(s,p) = TrialFunction(wSpace) # for linear solvers
(v,z) = TestFunction(wSpace)
#=====================;
#  Define parameters  ;
#=====================;
time = 0
class q_I_class(UserExpression):
    def eval(self, values, x):
        if 10 <= x[0] <= 20 and 10 <= x[1] <= 20:
            values[0] = 0.001
        else:
            values[0] = 0.0
# q_I = q_I_class()
qinj = q_I_class()
q_I = interpolate(qinj,kSpace)


class q_P_class(UserExpression):
    def eval(self, values, x):
        if 80 <= x[0] <= 90 and 80 <= x[1] <= 90:
            values[0] = 0.001
        else:
            values[0] = 0.0
# q_P = q_P_class()
qproj = q_P_class()
q_P = interpolate(qproj,kSpace)

# Initial condition
phi =  Expression("0.2", degree =3)
u = Function(wSpace)
u_k = Function(wSpace)
s_an =  Expression("0.15",t =time, degree =3)
p_an =  Expression("8500*(1-(x[0]+x[1]))",t =time, degree =3)
p0 = interpolate( p_an, pSpace)
s0 = interpolate( s_an, sSpace)
p_k = interpolate( p_an, pSpace)
s_k = interpolate( s_an, sSpace)
K = Constant(5e-8)
mu_w = Constant(5e-4)
mu_o = Constant(2e-3)

s_rw = 0.15
s_ro = 0.15

R = 0.05
P_d = 5000.
theta = 3.


def s_star(s):
    # return (s-s_rw)/(1-s_rw-s_ro)  #the numirator s_rw=0.15 gave numerical error 
    return s
def Ds_star_ds(s):
    # return 1./(1-s_rw-s_ro)
    return 1.

def Pc(s,s0):
    return conditional(gt(s0,R), P_d * s_star(s)**(-1./theta), P_d * R **(-1./theta) - P_d/theta * R **(-1-1./theta) * (s_star(s)-R))
def dPcds(s,s0):
    return conditional(gt(s0,R), P_d * s_star(s)**((-1./theta) -1) * (-1./theta) * (Ds_star_ds(s)),\
             - P_d/theta * R **(-1-1./theta ) * (Ds_star_ds(s)) )

# def Pc(s,s0):
#     return 5000 * s_star(s)**(-0.3333)

# def dPcds(s,s0):
#     return 5000 * (-0.3333) * s_star(s)**(-1.3333) * Ds_star_ds(s)

def eta_w(s):
    # return  s**(11./3.) * 1e-4
    return  s_star(s)**(11./3.) * 1e-4

def eta_o(s):
    # return (1-s)*(1-s)*(1-s**(11./3.)) * 2.5e-5
    return (1-s_star(s))*(1-s_star(s))*(1-s_star(s)**(11./3.)) * 2.5e-5

def f_w(s):
    return eta_w(s)/(eta_w(s)+eta_o(s))

def f_o(s):
    return eta_o(s)/(eta_w(s)+eta_o(s))
#=================================;
#  Dirichlet boundary conditions  ;
#=================================;
S_L = Constant(0.85)
def origin(x,on_boundary):
    return x[0] < DOLFIN_EPS and x[1] < DOLFIN_EPS 
bc_pressure_at_point = DirichletBC(wSpace.sub(1), Constant(4500), origin, 'pointwise')
bcs = [bc_pressure_at_point]
#============================;
#   Define variational form  ;
#============================;
n = FacetNormal(mesh)
h = CellDiameter(mesh)
h_avg = (h('+')+h('-'))/2
sigma = Constant(0.1)
##=====================================;
# Define the sSpace formulation  ;
#=====================================;
a_p = -(1./dt) * phi * s * z * dx +\
    eta_o(s_k) * inner(grad(p)+grad(dPcds(s0,s0)*s),grad(z)) * dx -\
    avg(inner(eta_o(s_k) * grad(p), n)) * jump(z) * dS -\
    avg(inner(eta_o(s_k) * grad(dPcds(s0,s0)*s), n)) * jump(z) * dS +\
    avg(inner(eta_o(s_k) * grad(z) , n )) * jump(p) * dS +\
    avg(inner(eta_o(s_k) * grad(z) , n )) * jump(dPcds(s0,s0)*s)* dS +\
    sigma/h_avg * jump(p) * jump(z)  * dS +\
    sigma/h_avg * jump(dPcds(s0,s0)*s) * jump(z)  * dS

L_p = - ((1./dt) * phi * s0 - (f_o(0.85) * q_I - f_o(s0)* q_P) ) * z * dx -\
    eta_o(s_k) * inner(grad(Pc(s0,s0)-dPcds(s0,s0)*s0),grad(z))*dx +\
    avg(inner(eta_o(s_k) * grad(Pc(s0,s0)-dPcds(s0,s0)*s0), n)) * jump(z) * dS -\
    avg(inner(eta_o(s_k) * grad(z) , n )) * jump(Pc(s0,s0)-dPcds(s0,s0)*s0)* dS -\
    sigma/h_avg * jump(Pc(s0,s0)-dPcds(s0,s0)*s0) * jump(z)  * dS


a_s = +(1./dt) * phi * s * v * dx +\
    eta_w(s_k) * inner(grad(p) , grad(v)) * dx -\
    avg(inner( eta_w(s_k) * grad(p) ,n)) * jump(v)  * dS +\
    avg(inner( eta_w(s_k) * grad(v) ,n)) * jump(p) * dS +\
    sigma/h_avg * jump(p) * jump(v)  * dS

L_s = +((1./dt) * phi * s0 + (f_w(0.85) * q_I - f_w(s0)* q_P) ) * v * dx

Bilinear = a_s + a_p
Linear = L_s + L_p
# ====================;
#  March over time   ;
#====================;
du = TrialFunction(wSpace)
file_s = File("./Output/sat_Picard.pvd")
file_p = File("./Output/pres_Picard.pvd")
file_error = File("./Output/error_DG%d_%d_.pvd")
time = 0
for n in range(num_steps):
    print ('time =%1.2f'% time)
    #update time
    time += dt
    s_an.time=time
    p_an.time=time

    eps = 1.0
    tol = 1.0e-5
    itr = 0
    maxiter = 250

    while eps > tol and itr < maxiter:
        itr += 1
        A_Bilinear = assemble(Bilinear)
        A_Linear = assemble(Linear)
        [bc.apply(A_Bilinear,A_Linear) for bc in bcs]

        # Set petsc options
        PETScOptions.set('ksp_view')
        PETScOptions.set('ksp_monitor_true_residual')
        PETScOptions.set('pc_type', 'fieldsplit')
        PETScOptions.set('pc_fieldsplit_type', 'additive')
        PETScOptions.set('pc_fieldsplit_detect_saddle_point')
        PETScOptions.set('fieldsplit_0_ksp_type', 'preonly')
        PETScOptions.set('fieldsplit_0_pc_type', 'lu')
        PETScOptions.set('fieldsplit_1_ksp_type', 'preonly')
        PETScOptions.set('fieldsplit_1_pc_type', 'jacobi')
        solver = PETScKrylovSolver()
        solver.set_operator(A_Bilinear)
        solver.ksp().setFromOptions()
        solver.solve(u.vector(),A_Linear)


        (sSol,pSol) = u.split(True)

        diff = sSol.vector().get_local()-s_k.vector().get_local()
        eps = np.linalg.norm(diff, ord=np.inf)
        s_k.assign(sSol)
        p_k.assign(pSol)


    Error = project(phi*(sSol-s0)/dt-div(eta_w(sSol)*grad(pSol))-(f_w(0.85) * q_I - f_w(s0)* q_P) ,ErrorSpace)
    Error.rename('Error','Error')

    #=====================;
    # Saving stiffness Mat
    #=====================;
    np.savetxt("./Mat/KDG_%d.dat"%time, A_Bilinear.array(),fmt='%3.3g',delimiter='\t')
    [r,c] = np.where(A_Bilinear.array()!=0)
    plt.scatter(c,-r,s=1,marker='*',c='r',label='Stiffness')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('./Mat/KDG_%d.png'%time)
    s0.assign(sSol)
    p0.assign(pSol)

    file_s << sSol,time
    file_p << pSol,time
    file_error.write(Error, time)
