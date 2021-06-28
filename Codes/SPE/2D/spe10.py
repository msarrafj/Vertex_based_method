from fenics import *
print(DirichletBC.__doc__)
import numpy as np
import time as tm
from petsc4py import PETSc
from scipy.io import loadmat
import math
import warnings
from ffc.quadrature.deprecation \
        import QuadratureRepresentationDeprecationWarning
warnings.simplefilter("ignore", QuadratureRepresentationDeprecationWarning)
parameters['reorder_dofs_serial'] = False # reorder dofs such that first s dofs and then p dofs

T = 86400.*2.5            # final time(second) 2.5 day
num_steps = 600     # number of time steps
dt = T / num_steps # time step size
#=====================================;
#  Create mesh and identify boundary  ;
#=====================================;
nx = 90
layer = '80' # three layers of 1, 45 and 80 are available
domain_length = 1000.
mesh = RectangleMesh(Point(0,0), Point(domain_length,domain_length), nx, nx)
boundaries = MeshFunction("size_t",mesh, mesh.topology().dim()-1)
#==========================;
#  Define function spaces ;
#=========================;
sSpace_FE = FiniteElement("CG", mesh.ufl_cell(),1)
pSpace_FE = FiniteElement("CG", mesh.ufl_cell(),1)
sSpace = FunctionSpace(mesh,sSpace_FE)
pSpace = FunctionSpace(mesh,pSpace_FE)
wSpace = FunctionSpace(mesh, sSpace_FE * pSpace_FE)
u = Function(wSpace)
kSpace = FunctionSpace(mesh,"DG",0)
M = len(wSpace.sub(0).dofmap().dofs()) #Size of dofs of either s or p
cell_domains = MeshFunction('size_t',mesh,2)
cell_domains.set_all(0)
element_s = wSpace.sub(0).element()
dofmap_s = wSpace.sub(0).dofmap()
element_p = wSpace.sub(1).element()
dofmap_p = wSpace.sub(1).dofmap()

counter = 0
for myCell in cells(mesh):
    cell_domains[myCell] = counter
    counter += 1

area = [cell.volume() for cell in cells(mesh)]
#===================================;
#  Define trial and test functions  ;
#===================================;
(s,p) = TrialFunction(wSpace) # for linear solvers
(z,v) = TestFunction(wSpace)

time = 0.
phi =  Expression("0.2", degree =3)
s_an =  Expression("0.15",t =time, degree =3)
p_an =  Expression("8500",t =time, degree =3)
p0 = interpolate( p_an, pSpace)
s0 = interpolate( s_an, sSpace)
# THis function import SPE 10 latyers that are extracted from MSRT in SI (m^2) unit
data = loadmat('./K_Input/Kx_%s.mat'%layer)
data_m = data['Kx_layer_%s'%layer]
def Perm(x,y):
    x_new , y_new = x*(60./domain_length) , y*(220./domain_length)
    # print(x_new,y_new)
    return data_m[math.floor(x_new),math.floor(y_new)]# x=60 and y=220 are out of bound   

mu_w = 5e-4
mu_o = 2e-3
s_rw = 0.15
s_ro = 0.15

def s_star(s):
    return (s-s_rw)/(1-s_rw-s_ro)

def Pc(s,s0):
    return 5000 * s**(-0.3333)
def dPcds(s,s0):
    return 5000 * (-0.3333) * s**(-1.3333)

def eta_w(s):
    return  s**(11./3.) * 1./mu_w

def eta_o(s):
    return (1-s)*(1-s)*(1-s**(11./3.)) * 1./mu_o

def f_w(s):
    return eta_w(s)/(eta_w(s)+eta_o(s))

def f_o(s):
    return eta_o(s)/(eta_w(s)+eta_o(s))

#=================================;
#  Dirichlet boundary conditions  ;
#=================================;
def origin(x,on_boundary):
    return x[0] < DOLFIN_EPS and x[1] < DOLFIN_EPS 
bc_pressure_at_point = DirichletBC(wSpace.sub(1), Constant(4500), origin, 'pointwise')

#====================;
#  March over time   ;
#====================;
u = Function(wSpace)
file_s = File("./Output/sat_vertexSPE10_layer_fine%s_.pvd"%layer)
file_p = File("./Output/pres_vertexSPE10_layer_fine%s_.pvd"%layer)
for n in range(num_steps):
    print ('time_step =%1.2f'% time)
    #update time
    time += dt
    dx = Measure('dx', subdomain_data = cell_domains)

    nrows = 2*M
    ncols = 2*M

    m_phi = np.zeros((nrows,ncols))
    etaC_IJ_B = np.zeros((nrows,ncols))
    etaC_IJ_D = np.zeros((nrows,ncols))
    etaC_IJPc = np.zeros((nrows,ncols))
    F_Term = np.zeros((nrows,1))
    E_Term = np.zeros((nrows,1))
    elnum = 0

    start_time = tm.time()
    C = np.matrix([[0.5,-0.5,0.0],[-0.5,1,-0.5],[0,-0.5,0.5]]) # only when dealing with structured mesh
    for cell in cells(mesh):
        # Define K
        K = Perm(cell.midpoint().x(),cell.midpoint().y())
        #Define q_I and q_P
        # note that int_\Omega q_I = int_\Omega q_P
        if 450 < cell.midpoint().x() < 550 and 450 < cell.midpoint().y() < 550:
            q_I = 0.28e-4
        else:
            q_I = 0.

        if (50 < cell.midpoint().x() < 150 and 50 < cell.midpoint().y() <150) or\
           (50 < cell.midpoint().x() < 150 and 850 < cell.midpoint().y() < 950) or\
           (850 < cell.midpoint().x() < 950 and 50 < cell.midpoint().y() < 150) or\
           (850 < cell.midpoint().x() < 950 and 850 < cell.midpoint().y() < 950):
            q_P = 0.28e-4/4.
        else:
            q_P = 0.
        dofs_s = dofmap_s.cell_dofs(cell.index())
        dofs_p = dofmap_p.cell_dofs(cell.index())
        ##==========================;
        ## Assemble Global A matrix ;
        ## =========================;
        for myrow in range(len(dofs_s)):
            rowLoc = dofs_s[myrow]
            m_phi[rowLoc,rowLoc] = m_phi[rowLoc,rowLoc] + 1./3. * area[elnum] * 0.2 / dt# for constant phi=0.2 only
        #==========================;
        # Assemble Global E vector ;
        # =========================;
            E_Term[rowLoc] = E_Term[rowLoc] + 1./3. * area[elnum] * 0.2 * s0.vector()[rowLoc] / dt#  It is eqv to ((1./dt) * phi * s0 ) * z * dx 
            E_Term[rowLoc] = E_Term[rowLoc] + 1./3. * area[elnum]* (f_w(0.85) * q_I - f_w(s0.vector()[rowLoc])* q_P) #m_i*qI_i
        #==========================;
        # Assemble Global B matrix ;
        # =========================;
            rowLoc_s = dofs_s[myrow]
            cij_eta_ij_B = 0
            rowLoc = dofs_p[myrow]
            cij_eta_ij_C = 0
            cij_eta_ij_D = 0
            cij_eta_ij_F = 0
            for mycol in [x for x in range(len(dofs_p)) if x != myrow]: # mycol \in {0,1,2} but != myrow
                colLoc = dofs_p[mycol]
                s_i = s0.vector()[rowLoc_s]
                s_j = s0.vector()[colLoc - M ] # [<>] should be global dof of s in matrix B (not Big matrix) 
                if p0.vector()[rowLoc_s]>p0.vector()[colLoc - M]:
                    value_etaij = eta_w(s_i) * K
                elif p0.vector()[rowLoc_s]<p0.vector()[colLoc-M]:
                    value_etaij = eta_w(s_j) * K
                elif p0.vector()[rowLoc_s]==p0.vector()[colLoc-M]:
                    value_etaij = eta_w(max(s_i,s_j)) * K
                else:
                    print('out of range')
                cij_eta_ij_B = cij_eta_ij_B + abs(C[myrow,mycol]) * value_etaij
                etaC_IJ_B[rowLoc_s,colLoc]= etaC_IJ_B[rowLoc_s,colLoc] - abs(C[myrow,mycol])* value_etaij
        #==========================;
        # Assemble Global C matrix ;
        # =========================;
                colLoc = dofs_s[mycol]
                s_i = s0.vector()[rowLoc - M]
                s_j = s0.vector()[colLoc]
                if p0.vector()[rowLoc - M]>p0.vector()[colLoc]:
                    value_etaij = eta_o(s_i) * K
                elif p0.vector()[rowLoc - M]<p0.vector()[colLoc]:
                    value_etaij = eta_o(s_j) * K
                elif p0.vector()[rowLoc - M]==p0.vector()[colLoc]:
                    value_etaij = eta_o(min(s_i,s_j)) * K
                else:
                    print('out of range')
                cij_eta_ij_C = cij_eta_ij_C + abs(C[myrow,mycol]) * value_etaij
                etaC_IJPc[rowLoc,colLoc] = etaC_IJPc[rowLoc,colLoc] - abs(C[myrow,mycol]) \
                                         * value_etaij * dPcds(s0.vector()[colLoc],s0.vector()[colLoc])
        #==========================;
        # Assemble Global D matrix ;
        # =========================;
                colLoc = dofs_p[mycol]
                s_i = s0.vector()[rowLoc - M]
                s_j = s0.vector()[colLoc - M ] # [<>] should be global dof of s in matrix B (not Big matrix) 
                if p0.vector()[rowLoc - M]>p0.vector()[colLoc - M]:
                    value_etaij = eta_o(s_i) * K
                elif p0.vector()[rowLoc - M]<p0.vector()[colLoc-M]:
                    value_etaij = eta_o(s_j) * K
                elif p0.vector()[rowLoc - M]==p0.vector()[colLoc-M]:
                    value_etaij = eta_o(min(s_i,s_j)) * K
                else:
                    print('out of range')
                cij_eta_ij_D = cij_eta_ij_D + abs(C[myrow,mycol]) * value_etaij
                etaC_IJ_D[rowLoc,colLoc]= etaC_IJ_D[rowLoc,colLoc] - abs(C[myrow,mycol])* value_etaij
        #==========================;
        # Assemble Global F matrix ;
        # =========================;
                colLoc = dofs_s[mycol] # 0 , 1, 2
                s_i = s0.vector()[rowLoc - M]
                s_j = s0.vector()[colLoc] # [<>] should be global dof of s in matrix B (not Big matrix) 
                if p0.vector()[rowLoc - M]>p0.vector()[colLoc]:
                    value_etaij = eta_o(s_i) * K
                elif p0.vector()[rowLoc - M]<p0.vector()[colLoc]:
                    value_etaij = eta_o(s_j) * K
                elif p0.vector()[rowLoc - M]==p0.vector()[colLoc]:
                    value_etaij = eta_o(min(s_i,s_j)) * K
                else:
                    print('out of range')
                cij_eta_ij_F = cij_eta_ij_F + abs(C[myrow,mycol]) * value_etaij

                F_Term[rowLoc] = F_Term[rowLoc] + abs(C[myrow,mycol]) \
                                         * value_etaij * ( - dPcds(s0.vector()[colLoc],s0.vector()[colLoc]) \
                                    * s0.vector()[colLoc]   + Pc(s0.vector()[colLoc],s0.vector()[colLoc]) )


            etaC_IJ_B[rowLoc_s,rowLoc_s+M] = etaC_IJ_B[rowLoc_s,rowLoc_s+M] + cij_eta_ij_B
            etaC_IJPc[rowLoc,rowLoc-M] = etaC_IJPc[rowLoc,rowLoc-M] + cij_eta_ij_C * dPcds(s0.vector()[rowLoc-M],s0.vector()[rowLoc-M])
            etaC_IJ_D[rowLoc,rowLoc] = etaC_IJ_D[rowLoc,rowLoc] + cij_eta_ij_D
            F_Term[rowLoc] = F_Term[rowLoc] + cij_eta_ij_F * ( dPcds(s0.vector()[rowLoc-M],s0.vector()[rowLoc-M]) \
                                    * s0.vector()[rowLoc-M] - Pc(s0.vector()[rowLoc-M],s0.vector()[rowLoc-M]))
            F_Term[rowLoc] = F_Term[rowLoc] - 1./3. * area[elnum] * 0.2 * s0.vector()[rowLoc-M] / dt#  It is eqv to -((1./dt) * phi * s0 ) * v * dx 
            F_Term[rowLoc] = F_Term[rowLoc] + 1./3. * area[elnum]* (f_o(0.85) * q_I - f_o(s0.vector()[rowLoc-M])* q_P) #m_i*qI_i
        elnum += 1
    core = tm.time()
    

    Evec = PETSc.Vec().create()
    Evec.setSizes(2*M)
    Evec.setUp()
    Evec.setValues(list(range(M-1)),E_Term[0:M-1])
    Emat_time = tm.time()

    Fvec = PETSc.Vec().create()
    Fvec.setSizes(2*M)
    Fvec.setUp()
    Fvec.setValues(list(range(M,nrows)),F_Term[M:(nrows+1)]) 
    Fmat_time = tm.time()

    LHSmat = PETSc.Mat().create()
    LHSmat.setSizes(2*M,2*M)
    LHSmat.setType("dense")
    LHSmat.setUp()
    LHSmat.setValues(list(range(M-1)),list(range(M)),m_phi[0:M-1,0:M]) # matrix A
    LHSmat.setValues(list(range(M-1)),list(range(M,ncols)),etaC_IJ_B[0:M-1,M:(ncols+1)]) # matrix B
    LHSmat.setValues(np.array([M-1],dtype=np.intc),list(range(M,ncols)),np.diag(m_phi[0:M,0:M])) # BoM conserved via matrix B last row
    LHSmat.setValues(list(range(M,nrows)),list(range(M)),etaC_IJPc[M:(nrows+1),0:M]-m_phi[0:M,0:M]) # matrix C
    LHSmat.setValues(list(range(M,nrows)),list(range(M,ncols)),etaC_IJ_D[M:(nrows+1),M:(ncols+1)])  # matrix D

    LHSmat.assemble()
    LHS_aij = LHSmat.convert("aij") 
    # We generated dense petsc matrices (as they are saving us tremendous time)
    # after index operations and assembly, we convert the LHS matrix to aij (or sparse matrix) this way we are able to 
    # solve problem with dolfin or petsc solver as they are desined for sparse matrices.:w
    # lgmap = PETSc.LGMap().create(wSpace.dofmap().dofs())
    # LHS_aij.setLGMap(lgmap, lgmap)
    # The above two lines help with establishing Dirichlet bcs, we then go ahead and bc_pressure_at_point.apply(LHS)
    LHS = PETScMatrix(LHS_aij)
    RHSvec = Evec+Fvec
    RHS = PETScVector(RHSvec)

    allLHS_time = tm.time()

    A_Bilinear = LHS
    A_Linear = RHS
    solve(A_Bilinear,u.vector(),A_Linear)
    solver_time = tm.time()

    (sSol,pSol) = u.split(True)
    s0.assign(sSol)
    p0.assign(pSol)
    file_s << sSol,time
    file_p << pSol,time
    print('core \t Evec_time \t Fvec_time \t LHS_assembly_time \t solver_time ')
    print('%2.4f \t %2.4f \t  %2.4f \t  %2.4f \t  %2.4f  \n'%(core-start_time, Emat_time-core, Fmat_time-Emat_time,allLHS_time-Fmat_time,solver_time-allLHS_time))
