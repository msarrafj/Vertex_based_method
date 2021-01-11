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

T = 86400.*0.1            # final time(second) 0.5 day
num_steps = 300     # number of time steps
dt = T / num_steps # time step size
#=====================================;
#  Create mesh and identify boundary  ;
#=====================================;
Lx = 50.
Ly = 100.
Lz = 24.
nx = 20
ny = 20
nz = 12
mesh = BoxMesh(Point(0.0, 0.0, 0.0), Point(Lx, Ly, Lz), nx, ny, nz)
# mesh= dolfin.Mesh('./mesh/3D/Unstructured_3D_fine.xml')
# mesh = RectangleMesh(Point(0,0), Point(domain_length,domain_length), nx, nx)
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
cell_domains = MeshFunction('size_t',mesh,3)
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
# (s,p) = (u[0],u[1]) # for non-linear solvers
(s,p) = TrialFunction(wSpace) # for linear solvers
(z,v) = TestFunction(wSpace)

time = 0.
phi =  Expression("0.2", degree =3)
s_an =  Expression("0.15",t =time, degree =3)
p_an =  Expression("8500",t =time, degree =3)
p0 = interpolate( p_an, pSpace)
s0 = interpolate( s_an, sSpace)
# THis function import SPE 10 latyers that are extracted from MSRT in SI (m^2) unit
data = loadmat('./K_Input/layers.mat')
data_m = data['Kx_layers']
def Perm(x,y,z):
    x_new , y_new , z_new = x*(32/Lx) , y*(64/Ly) , z*(12/Lz)
    return data_m[math.floor(x_new),math.floor(y_new),math.floor(z_new)]#unit m^2

# K = Constant(5e-827885)
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
# bc_pressure_at_point = DirichletBC(wSpace.sub(1), Constant(8500), origin)
# bcs = [bc_pressure_at_point]
# bcs = []

#====================;
#  March over time   ;
#====================;
u = Function(wSpace)
file_s = File("./Output/sat_vertexSPE10.pvd")
file_p = File("./Output/pres_vertexSPE10.pvd")
for n in range(num_steps):
    print ('time_step =%1.2f'% time)
    #update time
    time += dt
    dx = Measure('dx', subdomain_data = cell_domains)

    # mphi = np.zeros((40000,40000))
    nrows = 2*M
    ncols = 2*M

    print('nrows',nrows)
    m_phi = np.zeros((nrows,ncols))
    etaC_IJ_B = np.zeros((nrows,ncols))
    etaC_IJ_D = np.zeros((nrows,ncols))
    etaC_IJPc = np.zeros((nrows,ncols))
    F_Term = np.zeros((nrows,1))
    E_Term = np.zeros((nrows,1))
    elnum = 0

    start_time = tm.time()
    # C = np.matrix([[0.5,-0.5,0.0],[-0.5,1,-0.5],[0,-0.5,0.5]]) # only when dealing with structured mesh
    C = np.matrix([[0.01666667,-0.01666667,0.,0.],\
               [-0.01666667,0.03333333,-0.01666667,0.],\
               [ 0.,-0.01666667,0.03333333,-0.01666667],\
               [0.,0.,-0.01666667,0.01666667]])
    for cell in cells(mesh):
        # Define K
        # K = Perm(cell.midpoint().x(),cell.midpoint().y(),cell.midpoint().z())
        K = 1e-14
        #Define q_I and q_P
        # note that int_\Omega q_I = int_\Omega q_P = 2.
        if 5 < cell.midpoint().x() < 10 and 10 < cell.midpoint().y() < 20 and 18 < cell.midpoint().z() < 22:
            q_I = 0.01
        else:
            q_I = 0.

        if 40 < cell.midpoint().x() < 45 and 80 < cell.midpoint().y() < 90 and 18 < cell.midpoint().z() < 22:
            q_P = 0.01
        else:
            q_P = 0.
        # print("coordinate ele %s : \n %s "%(elnum,element_p.tabulate_dof_coordinates(cell))  )
        # C = assemble_local(inner(grad(p), grad(z))*dx(elnum),cell)[0:3,3:6]
        # print("C = ", C)
        dofs_s = dofmap_s.cell_dofs(cell.index())
        # print("dofs for s = \n",dofs_s)
        dofs_p = dofmap_p.cell_dofs(cell.index())
        # print("dofs for p = \n",dofs_p)
        ##==========================;
        ## Assemble Global A matrix ;
        ## =========================;
        for myrow in range(len(dofs_s)):
            rowLoc = dofs_s[myrow]
            m_phi[rowLoc,rowLoc] = m_phi[rowLoc,rowLoc] + 1./4. * area[elnum] * 0.2 / dt# for constant phi=0.2 only
        #==========================;
        # Assemble Global E vector ;
        # =========================;
            E_Term[rowLoc] = E_Term[rowLoc] + 1./4. * area[elnum] * 0.2 * s0.vector()[rowLoc] / dt#  It is eqv to ((1./dt) * phi * s0 ) * z * dx 
            E_Term[rowLoc] = E_Term[rowLoc] + 1./4. * area[elnum]* (f_w(0.85) * q_I - f_w(s0.vector()[rowLoc])* q_P) #m_i*qI_i
        #==========================;
        # Assemble Global B matrix ;
        # =========================;
        # for myrow in range(len(dofs_s)): # myrow is 0, 1, 2
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

            # etaC_IJ_B[rowLoc_s,rowLoc_s+M] = etaC_IJ_B[rowLoc_s,rowLoc_s+M] + cij_eta_ij_B
        #==========================;
        # Assemble Global C matrix ;
        # =========================;
        # for myrow in range(len(dofs_p)): # myrow is 3, 4, 5
            # rowLoc = dofs_p[myrow]
            # cij_eta_ij_C = 0
            # for mycol in [x for x in range(len(dofs_s)) if x != myrow]:
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
            # etaC_IJPc[rowLoc,rowLoc-M] = etaC_IJPc[rowLoc,rowLoc-M] + cij_eta_ij_C * dPcds(s0.vector()[rowLoc-M],s0.vector()[rowLoc-M])
        #==========================;
        # Assemble Global D matrix ;
        # =========================;
        # for myrow in range(len(dofs_p)): # myrow is 0, 1, 2
            # rowLoc = dofs_p[myrow] # 
            # cij_eta_ij_D = 0
            # for mycol in [x for x in range(len(dofs_p)) if x != myrow]:
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

            # etaC_IJ_D[rowLoc,rowLoc] = etaC_IJ_D[rowLoc,rowLoc] + cij_eta_ij_D
        #==========================;
        # Assemble Global F matrix ;
        # =========================;
        # for myrow in range(len(dofs_p)): # myrow is 3,4,5
            # rowLoc = dofs_p[myrow]
            # cij_eta_ij_F = 0
            # for mycol in [x for x in range(len(dofs_p)) if x != myrow]:
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
                                         # * value_etaij * ( - dPcds(s0.vector()[colLoc]) \
                                    * s0.vector()[colLoc]   + Pc(s0.vector()[colLoc],s0.vector()[colLoc]) )
                                    # * s0.vector()[colLoc]   + Pc(s0.vector()[colLoc])


            etaC_IJ_B[rowLoc_s,rowLoc_s+M] = etaC_IJ_B[rowLoc_s,rowLoc_s+M] + cij_eta_ij_B
            etaC_IJPc[rowLoc,rowLoc-M] = etaC_IJPc[rowLoc,rowLoc-M] + cij_eta_ij_C * dPcds(s0.vector()[rowLoc-M],s0.vector()[rowLoc-M])
            etaC_IJ_D[rowLoc,rowLoc] = etaC_IJ_D[rowLoc,rowLoc] + cij_eta_ij_D
            F_Term[rowLoc] = F_Term[rowLoc] + cij_eta_ij_F * ( dPcds(s0.vector()[rowLoc-M],s0.vector()[rowLoc-M]) \
                                    * s0.vector()[rowLoc-M] - Pc(s0.vector()[rowLoc-M],s0.vector()[rowLoc-M]))
            F_Term[rowLoc] = F_Term[rowLoc] - 1./4. * area[elnum] * 0.2 * s0.vector()[rowLoc-M] / dt#  It is eqv to -((1./dt) * phi * s0 ) * v * dx 
            F_Term[rowLoc] = F_Term[rowLoc] + 1./4. * area[elnum]* (f_o(0.85) * q_I - f_o(s0.vector()[rowLoc-M])* q_P) #m_i*qI_i
        elnum += 1
    core = tm.time()
    
    # Amat = PETSc.Mat().create()
    # Amat.setSizes(2*M,2*M)
    # Amat.setType("dense")
    # Amat.setUp()
    # Amat.setValues(list(range(M-1)),list(range(M)),m_phi[0:M-1,0:M])
    # # Amat.assemble()#

    # Amat_time = tm.time()

    # Bmat = PETSc.Mat().create()
    # Bmat.setSizes(2*M,2*M)
    # Bmat.setType("dense")
    # Bmat.setUp()
    # Bmat.setValues(list(range(M-1)),list(range(M,ncols)),etaC_IJ_B[0:M-1,M:(ncols+1)]) # not efficient we subt all the comps
    # Bmat.setValues(np.array([M-1],dtype=np.intc),list(range(M,ncols)),np.diag(m_phi[0:M,0:M])) # BoM conserved
    # # Bmat.assemble()#check if we need to add form_compiler_parameter in here?
    # Bmat_time = tm.time()

    # Cmat = PETSc.Mat().create()
    # Cmat.setSizes(2*M,2*M)
    # Cmat.setType("dense")
    # Cmat.setUp()
    # Cmat.setValues(list(range(M,nrows)),list(range(M)),etaC_IJPc[M:(nrows+1),0:M]-m_phi[0:M,0:M])
    # # Cmat.assemble()#check if we need to add form_compiler_parameter in here?
    # Cmat_time = tm.time()

    # Dmat = PETSc.Mat().create()
    # Dmat.setSizes(2*M,2*M)
    # Dmat.setType("dense")
    # Dmat.setUp()
    # Dmat.setValues(list(range(M,nrows)),list(range(M,ncols)),etaC_IJ_D[M:(nrows+1),M:(ncols+1)]) # not efficient we subt all the comps
    # # Dmat.assemble()
    # Dmat_time = tm.time()

    Evec = PETSc.Vec().create()
    Evec.setSizes(2*M)
    Evec.setUp()
    Evec.setValues(list(range(M-1)),E_Term[0:M-1])
    # Evec.assemble()
    Emat_time = tm.time()

    Fvec = PETSc.Vec().create()
    Fvec.setSizes(2*M)
    Fvec.setUp()
    Fvec.setValues(list(range(M,nrows)),F_Term[M:(nrows+1)]) 
    # Fvec.assemble()
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

    # LHSmat = Amat+Bmat+Cmat+Dmat
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
    # bc_pressure_at_point.apply(LHS)

    allLHS_time = tm.time()

    A_Bilinear = LHS
    A_Linear = RHS
    # [bc.apply(A_Bilinear,A_Linear) for bc in bcs]
    # Set petsc options
    # PETScOptions.set("ksp_type", "fgmres")
    # PETScOptions.set("ksp_rtol", 1e-4)
    # PETSc.Options()['ksp_max_it'] = 1000000
    # PETScOptions.set('pc_type', 'fieldsplit')
    # PETScOptions.set('pc_fieldsplit_type', 'schur')
    # PETScOptions.set('pc_fieldsplit_0_fields', '0')
    # PETScOptions.set('pc_fieldsplit_1_fields', '1')
    # PETScOptions.set('pc_fieldsplit_detect_saddle_point')
    # PETScOptions.set('pc_fieldsplit_shur_face_type', 'upper')
    # PETScOptions.set('fieldsplit_0_ksp_type', 'preonly')
    # PETScOptions.set('fieldsplit_0_pc_type', 'sor')
    # PETScOptions.set('fieldsplit_1_ksp_type', 'gmres')
    # PETScOptions.set('fieldsplit_1_pc_type', 'hypre')
    # PETScOptions.set('fieldsplit_1_pc_hypre_type', 'boomeramg')
    # PETScOptions.set('pc_fieldsplit_schur_precondition','selfp')

    # solver = PETScKrylovSolver()
    # solver.set_operator(A_Bilinear)
    # solver.ksp().setFromOptions()
    # # # Solve the problem
    # solver.solve(u.vector(),A_Linear)
    # solver.solve(A_Bilinear,u.vector(),A_Linear)
    solve(A_Bilinear,u.vector(),A_Linear)
    solver_time = tm.time()

    (sSol,pSol) = u.split(True)
    s0.assign(sSol)
    p0.assign(pSol)
    file_s << sSol,time
    file_p << pSol,time
    print('core \t Evec_time \t Fvec_time \t LHS_assembly_time \t solver_time ')
    print('%2.4f \t %2.4f \t  %2.4f \t  %2.4f \t  %2.4f  \n'%(core-start_time, Emat_time-core, Fmat_time-Emat_time,allLHS_time-Fmat_time,solver_time-allLHS_time))
