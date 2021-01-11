from fenics import *
from mshr import *
import numpy as np
import time as tm
import math,sys,time
from petsc4py import PETSc
import warnings
from ffc.quadrature.deprecation \
        import QuadratureRepresentationDeprecationWarning
warnings.simplefilter("ignore", QuadratureRepresentationDeprecationWarning)
parameters['reorder_dofs_serial'] = False # reorder dofs such that first s dofs and then p dofs

nx = int(sys.argv[1])
num_steps = int(sys.argv[2])
CASE = sys.argv[3]

T = 1.            # final time
dt = T / num_steps # time step size
#=====================================;
#  Create mesh and identify boundary  ;
#=====================================;
mesh = UnitSquareMesh(nx,nx)
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
u_k = Function(wSpace)
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

# Initial condition
time = 0.
if CASE == 'Case4':
    #----------CaseIV----------
    phi =  Expression("0.2", degree =3)
    s_an =  Expression("0.2*(2.+2.*x[0]*x[1]+cos(time+x[0]))",time = time, degree =3) 
    p_an =  Expression("2.+pow(x[0],2)*x[1]-pow(x[1],2)+pow(x[0],2)*sin(time+x[1])\
               -1./6. * ( 2*cos(time) -2*cos(time+1) + 11 )",time = time, degree =3)
    p0 = interpolate( p_an, pSpace)
    s0 = interpolate( s_an, sSpace)
    p_k = interpolate( p_an, pSpace)
    s_k = interpolate( s_an, sSpace)
    def Pc(s,s0):
        return conditional(gt(s0,0.05), 50 * s **(-0.5), 50* (1.5-10*s)*0.05**(-0.5))
    def dPcds(s,s0):
        return conditional(gt(s0,0.05), -25 * s **(-1.5), -500*0.05**(-0.5) )
    # NON-UFL Pc function that could be used as boolean in python
    def Pc_none(s,s0):
        if (s0>0.05):
            return  50 * s **(-0.5)
        else:
            return 50* (1.5-10*s)*0.05**(-0.5)

    def eta_w(s):
        return  s*s
    def eta_o(s):
        return (1-s)*(1-s)

    q_w = Expression("-1* phi* (0.2*sin(time+x[0]))\
     -( 0.04*pow(2. + 2.*x[0]*x[1] + cos(time + x[0]),2)*(2*x[1] + 2*sin(time + x[1])) + \
     0.08*(2. + 2.*x[0]*x[1] + cos(time + x[0]))*(2.*x[1] - sin(time + x[0]))*(2*x[0]*x[1] + 2*x[0]*sin(time + x[1]))\
     +.16*x[0]*(2. + 2.*x[0]*x[1] + cos(time + x[0]))*(pow(x[0],2) - 2*x[1] + pow(x[0],2)*cos(time + x[1])) +\
     0.04*pow(2. + 2.*x[0]*x[1] + cos(time + x[0]),2)*(-2 - pow(x[0],2)*sin(time + x[1]))\
     )",phi=phi,time=time,degree=3)

    q_o = Expression("phi*0.2*sin(time+x[0])  \
     -(pow(1 - 0.2*(2. + 2.*x[0]*x[1] + cos(time + x[0])),2)*(2*x[1] + 2*sin(time + x[1])) -\
         0.4*(1 - 0.2*(2. + 2.*x[0]*x[1] + cos(time + x[0])))*(2.*x[1] - sin(time + x[0]))*(2*x[0]*x[1] + 2*x[0]*sin(time + x[1]))\
         -0.8*x[0]*(1 - 0.2*(2. + 2.*x[0]*x[1] + cos(time + x[0])))*(pow(x[0],2) - 2*x[1] + pow(x[0],2)*cos(time + x[1])) +\
         pow(1 - 0.2*(2. + 2.*x[0]*x[1] + cos(time + x[0])),2)*(-2 - pow(x[0],2)*sin(time + x[1]))\
         )\
     +(\
         (1.118033988749895*(0. - cos(time + x[0]))*pow(1 - 0.2*(2. + 2.*x[0]*x[1] + cos(time + x[0])),  2))/ \
         pow(2. + 2.*x[0]*x[1] + cos(time + x[0]),1.5) - \
         (0.447213595499958*(1 - 0.2*(2. + 2.*x[0]*x[1] + cos(time + x[0])))*pow(2.*x[1] - sin(time + x[0]),2))/ \
         pow(2. + 2.*x[0]*x[1] + cos(time + x[0]),1.5) - \
         (1.6770509831248424*pow(1 - 0.2*(2. + 2.*x[0]*x[1] + cos(time + x[0])),2)*pow(2.*x[1] - \
         sin(time + x[0]),2))/ \
         pow(2. + 2.*x[0]*x[1] + cos(time + x[0]),2.5) \
         +(-1.788854381999832*pow(x[0],2)*(1 - 0.2*(2. + 2.*x[0]*x[1] + cos(time + x[0]))))/ \
         pow(2. + 2.*x[0]*x[1] + cos(time + x[0]),1.5) - \
         (6.708203932499369*pow(x[0],2)*pow(1 - 0.2*(2. + 2.*x[0]*x[1] + cos(time + x[0])),2))/ \
         pow(2. + 2.*x[0]*x[1] + cos(time + x[0]),2.5) \
         )*50",phi=phi,time=time,degree=3 )
else:
    print("imported case not found!")
#=================================;
#  Dirichlet boundary conditions  ;
#=================================;
bc_s = DirichletBC(wSpace.sub(0),s_an,"on_boundary")
bc_p = DirichletBC(wSpace.sub(1),p_an,"on_boundary")
bcs = [bc_s , bc_p]
#====================;
#  March over time   ;
#====================;
file_s = File("./Output/sat_vertex.pvd")
file_sEx = File("./Output/s_exact.pvd")
file_p = File("./Output/pres_vertex.pvd")
file_pEx = File("./Output/p_exact.pvd")
for n in range(num_steps):
    #update time
    time += dt
    print('time_step = %1.2f'% time)
    q_w.time = time
    q_o.time = time
    s_an.time=time
    p_an.time=time
    s_ex = interpolate(s_an,sSpace)
    p_ex = interpolate(p_an,pSpace)
    p_ex.rename("p_ex","p_ex")
    s_ex.rename("s_ex","s_ex")
    # file_sEx << s_ex,time
    # file_pEx << p_ex,time
    # Picard parameters
    eps = 1.0
    tol = 1.0e-6
    itr = 0
    maxiter = 250

    while eps > tol and itr < maxiter:
        itr += 1
        dx = Measure('dx', subdomain_data = cell_domains)

        nrows = 2*M
        ncols = 2*M

        m_phi = np.zeros((nrows,ncols))
        etaC_IJ_B = np.zeros((nrows,ncols))
        etaC_IJ_D = np.zeros((nrows,ncols))
        etaC_IJPc = np.zeros((nrows,ncols))
        F_Term = np.zeros((nrows,1))
        E_Term = np.zeros((nrows,1))
        E = assemble(+ ((1./dt) * phi * s0 ) * z * dx +( q_w) * z * dx
        , form_compiler_parameters={\
            'quadrature_degree': 1,'quadrature_rule': 'vertex','representation': 'quadrature'})

        Evec =  as_backend_type(E).vec()#copy vector 
        Evec.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

        Fvec =  as_backend_type(E).vec().duplicate()#generate zeros vector
        Fvec.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        elnum = 0

        start_time = tm.time()
        C = np.matrix([[0.5,-0.5,0.0],[-0.5,1,-0.5],[0,-0.5,0.5]]) # only when dealing with structured mesh
        for cell in cells(mesh):
            # C = assemble_local(inner(grad(p), grad(z))*dx(elnum),cell)[0:3,3:6]
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
                # E_Term[rowLoc] = E_Term[rowLoc] + 1./3. * area[elnum] * 0.2 * s0.vector()[rowLoc] / dt#  It is eqv to ((1./dt) * phi * s0 ) * z * dx 
                # E_Term[rowLoc] = E_Term[rowLoc] + 1./3. * area[elnum]*  q_w(s0.vector()[rowLoc]) #m_i*qI_i
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
                    s_i = s_k.vector()[rowLoc_s]
                    s_j = s_k.vector()[colLoc - M ] # [<>] should be global dof of s in matrix B (not Big matrix) 
                    if p_k.vector()[rowLoc_s]>p_k.vector()[colLoc - M]:
                        value_etaij = eta_w(s_i)
                    elif p_k.vector()[rowLoc_s]<p_k.vector()[colLoc-M]:
                        value_etaij = eta_w(s_j)
                    elif p_k.vector()[rowLoc_s]==p_k.vector()[colLoc-M]:
                        value_etaij = eta_w(max(s_i,s_j))
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
                    s_i = s_k.vector()[rowLoc - M]
                    s_j = s_k.vector()[colLoc]
                    po_i = (p_k.vector()[rowLoc - M] + Pc_none(s_i,s_i))
                    po_j = (p_k.vector()[colLoc] + Pc_none(s_j,s_j) )
                    if po_i > po_j :
                        value_etaij = eta_o(s_i)
                    elif po_i < po_j:
                        value_etaij = eta_o(s_j)
                    elif po_i == po_j:
                        value_etaij = eta_o(min(s_i,s_j))
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
                    s_i = s_k.vector()[rowLoc - M]
                    s_j = s_k.vector()[colLoc - M ] # [<>] should be global dof of s in matrix B (not Big matrix) 
                    po_i = p_k.vector()[rowLoc - M] + Pc_none(s_i,s_i)
                    po_j = p_k.vector()[colLoc - M] + Pc_none(s_j,s_j)
                    if po_i > po_j :
                        value_etaij = eta_o(s_i)
                    elif po_i < po_j:
                        value_etaij = eta_o(s_j)
                    elif po_i == po_j:
                        value_etaij = eta_o(min(s_i,s_j))
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
                    s_i = s_k.vector()[rowLoc - M]
                    s_j = s_k.vector()[colLoc] # [<>] should be global dof of s in matrix B (not Big matrix) 
                    po_i = p_k.vector()[rowLoc - M] + Pc_none(s_i,s_i)
                    po_j = p_k.vector()[colLoc] + Pc_none(s_j,s_j)
                    if po_i > po_j :
                        value_etaij = eta_o(s_i)
                    elif po_i < po_j:
                        value_etaij = eta_o(s_j)
                    elif po_i == po_j:
                        value_etaij = eta_o(min(s_i,s_j))
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
                # F_Term[rowLoc] = F_Term[rowLoc] - 1./3. * area[elnum] * 0.2 * s0.vector()[rowLoc-M] / dt#  It is eqv to -((1./dt) * phi * s0 ) * v * dx 
                # F_Term[rowLoc] = F_Term[rowLoc] + 1./3. * area[elnum]* q_o(s0.vector()[rowLoc-M]) #m_i*qI_i
            elnum += 1
        core = tm.time()

        Evec.setValues(list(range(M-1,M)),0)
        Evec.assemble()
        E = PETScVector(Evec)
        # np.savetxt("E_old.dat", E[:], fmt='%3.3g', delimiter= '\t' )
        Emat_time = tm.time()

        Fvec.setValues(list(range(M,nrows)),F_Term[M:(nrows+1)]) # not efficient we subt all the comps
        Fvec.assemble()#check if we need to add form_compiler_parameter in here?
        F_mass = assemble(- ((1./dt) * phi * s0) * v * dx + (q_o)  * v * dx , form_compiler_parameters={\
            'quadrature_degree': 1,'quadrature_rule': 'vertex','representation': 'quadrature'}) # mass lumped
        F = PETScVector(Fvec) + F_mass
        # np.savetxt("F_old.dat", F[:], fmt='%3.3g', delimiter= '\t' )
        # Fvec = PETSc.Vec().create()
        # Fvec.setSizes(2*M)
        # Fvec.setUp()
        # Fvec.setValues(list(range(M,nrows)),F_Term[M:(nrows+1)]) 
        # # Fvec.assemble()
        # np.savetxt("F_new.dat", Fvec[:], fmt='%3.3g', delimiter= '\t' )
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
        # solve problem with dolfin or petsc solver as they are desined for sparse matrices.
        lgmap = PETSc.LGMap().create(wSpace.dofmap().dofs())
        LHS_aij.setLGMap(lgmap, lgmap)
        # The above two lines help with establishing Dirichlet bcs, we then go ahead and bc_pressure_at_point.apply(LHS)
        LHS = PETScMatrix(LHS_aij)
        # RHSvec = Evec+Fvec
        # RHS = PETScVector(RHSvec)
        RHS = F+E # old RHS

        allLHS_time = tm.time()

        A_Bilinear = LHS
        A_Linear = RHS
        bc_s.apply(A_Bilinear,A_Linear)
        bc_p.apply(A_Bilinear,A_Linear)
        # [bc.apply(A_Bilinear,A_Linear) for bc in bcs]

        solve(A_Bilinear,u.vector(),A_Linear)
        solver_time = tm.time()
        (sSol,pSol) = u.split(True)

        diff = sSol.vector().get_local()-s_k.vector().get_local()
        eps = np.linalg.norm(diff, ord=np.inf)
        print('iter=%d: norm=%g'%(itr,eps))
        s_k.assign(sSol)
        p_k.assign(pSol)


    s0.assign(sSol)
    p0.assign(pSol)

    # file_s << sSol,time
    # file_p << pSol,time
    # print('core \t Evec_time \t Fvec_time \t LHS_assembly_time \t solver_time ')
    # print('%2.4f \t %2.4f \t  %2.4f \t  %2.4f \t  %2.4f  \n'%(core-start_time, Emat_time-core, Fmat_time-Emat_time,allLHS_time-Fmat_time,solver_time-allLHS_time))
#=================;
#Exact solution   ;
#=================;
L2_error_s = errornorm(s_ex,s0,norm_type='L2',degree_rise= 5)
L2_error_p = errornorm(p_ex,p0,norm_type='L2',degree_rise= 5)
print("L2_error_of_s %e" % L2_error_s)
print("L2_error_of_p %e" % L2_error_p)
H1_error_s = errornorm(s_ex,s0,norm_type='H10',degree_rise= 5)
H1_error_p = errornorm(p_ex,p0,norm_type='H10',degree_rise= 5)
print("H1_error_of_s %e" % H1_error_s)
print("H1_error_of_p %e" % H1_error_p)
