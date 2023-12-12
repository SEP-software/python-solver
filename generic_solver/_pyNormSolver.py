# Module containing Norm Solver classes
from math import isnan
import numpy as np
prec= 1e-8

zero = 10 ** (np.floor(np.log10(np.abs(float(np.finfo(np.float64).tiny)))) + 2)  # Check for avoid Overflow or Underflow
from generic_solver._pySolver import Solver
from generic_solver._pyProblem import ProblemLinearSymmetric
class Normsolver(Solver):
    """Linear-Conjugate Gradient and Steepest-Descent Solver parent object"""

    # Default class methods/functions
    def __init__(self, stopper, steepest=False, logger=None, ntrys=8):
        """
        Constructor for LCG/SD Solver:
        :param stopper: Stopper, object to terminate inversion
        :param steepest: bool, use the steepest-descent instead of conjugate gradient [False]
        :param logger: Logger, object to write inversion log file [None]
        :param ntrys: Number of tries to reduce 
        """
        # Calling parent construction
        super(LCGsolver, self).__init__()
        # Defining stopper object
        self.stopper = stopper
        # Whether to run steepest descent or not
        self.steepest = steepest
        # Logger object to write on log file
        self.logger = logger
        # Overwriting logger of the Stopper object
        self.stopper.logger = self.logger
        # Number of tries to reduce cost functon
        self._ntrys=ntrys
        # print formatting
        self.iter_msg = "iter = %s, obj = %.5e, resnorm = %.2e, gradnorm = %.2e, feval = %d"

    def __del__(self):
        """Default destructor"""
        return

    def run(self, problem, verbose=False, restart=False):
        """Running LCG and steepest-descent solver"""
        self.create_msg = verbose or self.logger

        # Resetting stopper before running the inversion
        self.stopper.reset()
        # Check for preconditioning
        precond = True if "prec" in dir(problem) and problem.prec is not None else False

        if not restart:
            if self.create_msg:
                msg = 90 * "#" + "\n"
                msg += "\t\t\t\tPRECONDITIONED " if precond else "\t\t\t\t"
                msg += "LINEAR %s SOLVER\n" % ("STEEPEST-DESCENT" if self.steepest else "CONJUGATE GRADIENT log file")
                msg += "\tRestart folder: %s\n" % self.restart.restart_folder
                msg += "\tModeling Operator:\t\t%s\n" % problem.op
                msg += 90 * "#" + "\n"
                if verbose:
                    print(msg.replace("log file", ""))
                if self.logger:
                    self.logger.addToLog(msg)

            # Setting internal vectors (model and search direction vectors)
            prblm_mdl = problem.get_model()
            cg_mdl = prblm_mdl.clone()
            cg_dmodl = prblm_mdl.clone().zero()

            # Other internal variables
            iiter = 0
        else:
            # Retrieving parameters and vectors to restart the solver
            if self.create_msg:
                msg = "Restarting previous solve run from: %s" % self.restart.restart_folder
                if verbose:
                    print(msg)
                if self.logger:
                    self.logger.addToLog(msg)
            self.restart.read_restart()
            iiter = self.restart.retrieve_parameter("iter")
            initial_obj_value = self.restart.retrieve_parameter("obj_initial")
            cg_mdl = self.restart.retrieve_vector("cg_mdl")
            cg_dmodl = self.restart.retrieve_vector("cg_dmodl")
            if not precond:
                cg_dres = self.restart.retrieve_vector("cg_dres")
            else:
                dot_grad_prec_grad = self.restart.retrieve_vector("dot_grad_prec_grad")
            # Setting the model and residuals to avoid residual twice computation
            problem.set_model(cg_mdl)
            prblm_mdl = problem.get_model()
            # Setting residual vector to avoid its unnecessary computation
            problem.set_residual(self.restart.retrieve_vector("prblm_res"))

        # Common variables unrelated to restart
        success = True
        # Variables necessary to return inverted model if inversion stops earlier
        prev_mdl = prblm_mdl.clone().zero()
        if precond:
            cg_prec_grad = cg_dmodl.clone().zero()


        itry=0
        
        # Iteration loop
        while True:
            # Computing objective function
            prblm_res = problem.get_res(cg_mdl)  # Compute residuals
            obj0 = prblm_res.norm() # Compute objective function value
            prblm_grad = problem.get_grad(cg_mdl)  # Compute the gradient
            c0,c1,c2 = problem_res.calcC0C1C2()
            if iiter == 0:
                initial_obj_value = obj0  # For relative objective function value
                # Saving initial objective function value
                self.restart.save_parameter("obj_initial", initial_obj_value)
                if self.create_msg:
                    msg = self.iter_msg % (str(iiter).zfill(self.stopper.zfill),
                                           obj0,
                                           problem.get_rnorm(cg_mdl),
                                           problem.get_gnorm(cg_mdl),
                                           problem.get_fevals())
                    if verbose:
                        print(msg)
                    # Writing on log file
                    if self.logger:
                        self.logger.addToLog(msg)
                # Check if either objective function value or gradient norm is NaN
                if isnan(obj0) or isnan(prblm_grad.norm()):
                    raise ValueError("Either gradient norm or objective function value NaN!")
                # Set internal delta residual vector
                if not precond:
                    cg_dres = prblm_res.clone().zero()
            if prblm_grad.norm() == 0.:
                print("Gradient vanishes identically")
                break

            # Saving results
            self.save_results(iiter, problem, force_save=False)
            prev_mdl.copy(prblm_mdl)  # Keeping the previous model

            # Computing alpha and beta coefficients
            if precond:
                # Applying preconditioning to current gradient
                problem.prec.forward(False, prblm_grad, cg_prec_grad)
                if iiter == 0 or self.steepest:
                    # Steepest descent
                    beta = 0.
                    dot_grad_prec_grad = prblm_grad.dot(cg_prec_grad)
                else:
                    # Conjugate-gradient coefficients for preconditioned CG
                    dot_grad_prec_grad_old = dot_grad_prec_grad
                    if dot_grad_prec_grad_old == 0.:
                        success = False
                        # Writing on log file
                        if self.logger:
                            self.logger.addToLog("Gradient orthogonal to preconditioned one, will terminate solver")
                    dot_grad_prec_grad = prblm_grad.dot(cg_prec_grad)
                    beta = dot_grad_prec_grad / dot_grad_prec_grad_old
                # Update search direction
                cg_dmodl.scaleAdd(cg_prec_grad, beta, 1.0)
                cg_dmodld = problem.get_dres(cg_mdl, cg_dmodl)  # Project search direction into the data space
                dot_cg_dmodld = cg_dmodld.dot(cg_dmodld)
                if dot_cg_dmodld == 0.0:
                    success = False
                    # Writing on log file
                    if self.logger:
                        self.logger.addToLog(
                            "Search direction orthogonal to span of linear operator, will terminate solver")
                else:
                    alpha = - dot_grad_prec_grad / dot_cg_dmodld
                    # Writing on log file
                    if beta == 0.:
                        msg = "Steepest-descent step length: %.2e" % alpha
                    else:
                        msg = "Conjugate alpha, beta: %.2e, %.2e" % (alpha, beta)
                    if self.logger:
                        self.logger.addToLog(msg)
            else:
                prblm_gradd = problem.get_dres(cg_mdl, prblm_grad)  # Project gradient into the data space
                # Computing alpha and beta coefficients
                if iiter == 0 or self.steepest:
                    # Steepest descent
                    beta = 0.0
                    dot_gradd = prblm_gradd.dot(prblm_gradd)
                    if dot_gradd <= zero:
                        success = False
                        # Writing on log file
                        if self.logger:
                            self.logger.addToLog(
                                "Gradient orthogonal to span of linear operator, will terminate solver")
                    else:
                        dot_gradd_res = prblm_gradd.dot(c2)
                        alpha = - np.real(dot_gradd_res) / dot_gradd
                        msg = "Steppest-descent step length: " + str(alpha)
                        # Writing on log file
                        if iiter == 0:
                            msg = "First steppest-descent step length: " + str(alpha)
                        if self.logger:
                            self.logger.addToLog(msg)
                else:
                    # Conjugate-gradient coefficients
                    dot_gradd = prblm_gradd.dot(prblm_gradd)
                    dot_dres = cg_dres.dot(cg_dres)
                    dot_gradd_dres = np.real(prblm_gradd.dot(cg_dres))
                    if dot_gradd <= zero or dot_dres <= zero:
                        success = False
                    else:
                        determ = dot_gradd * dot_dres - dot_gradd_dres * dot_gradd_dres
                        # Checking if alpha or beta are becoming infinity
                        if abs(determ) < zero:
                            if self.create_msg:
                                msg = "Plane-search method fails (zero det: %.2e), will terminate solver" % determ
                                if verbose:
                                    print(msg)
                                if self.logger:
                                    self.logger.addToLog(msg)
                            break
                        c1dg=-dot_gradd.dot(c1);  //c1dg = -C'*G
                        c1ds=-ss.dot(c1);
                        dot_gradd_res = np.real(prblm_gradd.dot(prblm_res))
                        dot_dres = np.real(cg_dres.dot(prblm_res))
                        c2dgg=c2.dot(dot_gradd);//c2dgg = C"*G*G
                        c2dss=c2.dot(dot_dres);//c2dss = C"*S*S
                        c2dgs=c2.dot(dot_gradd_dres);//c2dgs = C"*G*S
                        determ=1-(c2dgs/c2dgg)*(c2dgs/c2dss);
                        determ = c2dgg * c2dss * determ;
                        alpha = ( c2dss * c1dg - c2dgs * c1ds ) / determ;
                        beta = (-c2dgs * c1dg + c2dgg * c1ds ) / determ;
                        # Writing on log file
                        if self.logger:
                            self.logger.addToLog("Conjugate alpha,beta: " + str(alpha) + ", " + str(beta))
        itry=0
        found=False
        cg_mdl_save=cg_mdl.clone()
        while itry < self.ntrys:
            if not success:
                if self.create_msg:
                    msg = "Stepper couldn't find a proper step size, will terminate solver"
                    if verbose:
                        print(msg)
                    if self.logger:
                        self.logger.addToLog(msg)
                break

            if precond:
                # modl = modl + alpha * dmodl
                cg_mdl.scaleAdd(cg_dmodl, 1.0, alpha)  # Update model
            else:
                # dmodl = alpha * grad + beta * dmodl
                cg_dmodl.scaleAdd(prblm_grad, beta, alpha)  # update search direction
                # modl = modl + dmodl
                cg_mdl.scaleAdd(cg_dmodl)  # Update model

            # Setting the model
            problem.set_model(cg_mdl)
            # Projecting model onto the bounds (if any)
            if "bounds" in dir(problem):
                problem.bounds.apply(cg_mdl)

            if prblm_mdl.isDifferent(cg_mdl):
                # Model went out of the bounds
                msg = "Model hit provided bounds. Projecting it onto them."
                if self.logger:
                    self.logger.addToLog(msg)
                # Recomputing m_current = m_new - dmodl
                prblm_mdl.scaleAdd(cg_dmodl, 1.0, -1.0)
                # Finding the projected dmodl = m_new_clipped - m_current
                cg_dmodl.copy(cg_mdl)
                cg_dmodl.scaleAdd(prblm_mdl, 1.0, -1.0)
                problem.set_model(cg_mdl)
                if precond:
                    cg_dmodl.scale(1.0 / alpha)  # Unscaling the search direction
                else:
                    # copying previous residuals dres = res_old
                    cg_dres.copy(prblm_res)
                    # Computing actual change in the residual vector dres = res_new - res_old
                    prblm_res = problem.get_res(cg_mdl)  # New residual vector
                    cg_dres.scaleAdd(prblm_res, -1.0, 1.0)
            else:
                # Setting residual vector to avoid its unnecessary computation (if model was not clipped)
                if precond:
                    # res = res + alpha * dres
                    prblm_res.scaleAdd(cg_dmodld, 1.0, alpha)  # Update residuals
                else:
                    # dres  = alpha * gradd + beta * dres
                    cg_dres.scaleAdd(prblm_gradd, beta, alpha)  # Update residual step
                    # res = res + dres
                    prblm_res.scaleAdd(cg_dres)  # Update residuals
                problem.set_residual(prblm_res)

            obj1 = problem.get_res(cg_mdl)
            if obj1 < obj0:
                if self.create_msg:
                    msg = "Objective function didn't reduce, reducing alpha/beta:\n\t" \
                            "obj_new = %.5e\tobj_cur = %.5e" % (obj1, obj0)
                if verbose:
                    print(msg)
                alpha=alpha/2.
                beta=beta/2.
                cg_mdl=cg_mdl_save.clone()
                itry+=1
            else:
                found=True            
            # Increasing iteration counter
            iiter += 1
            # Computing new objective function value
            obj1 = problem.get_res(cg_mdl)
            if obj1 >= obj0:
                if self.create_msg:
                    msg = "Objective function didn't reduce, will terminate solver:\n\t" \
                          "obj_new = %.5e\tobj_cur = %.5e" % (obj1, obj0)
                    if verbose:
                        print(msg)
                    # Writing on log file
                    if self.logger:
                        self.logger.addToLog(msg)
                problem.set_model(prev_mdl)
                break

            # Saving current model and previous search direction in case of restart
            self.restart.save_parameter("iter", iiter)
            self.restart.save_vector("cg_mdl", cg_mdl)
            self.restart.save_vector("cg_dmodl", cg_dmodl)
            # Saving data space vectors or scaling if preconditioned
            if not precond:
                self.restart.save_vector("cg_dres", cg_dres)
            else:
                self.restart.save_parameter("dot_grad_prec_grad", dot_grad_prec_grad)
            self.restart.save_vector("prblm_res", prblm_res)

            # iteration info
            if self.create_msg:
                msg = self.iter_msg % (str(iiter).zfill(self.stopper.zfill),
                                       obj1,
                                       problem.get_rnorm(cg_mdl),
                                       problem.get_gnorm(cg_mdl),
                                       problem.get_fevals())
                if verbose:
                    print(msg)
                # Writing on log file
                if self.logger:
                    self.logger.addToLog("\n" + msg)
            # Check if either objective function value or gradient norm is NaN
            if isnan(obj1) or isnan(prblm_grad.norm()):
                raise ValueError("Either gradient norm or objective function value NaN!")
            if self.stopper.run(problem, iiter, initial_obj_value, verbose):
                break

        # Writing last inverted model
        self.save_results(iiter, problem, force_save=True, force_write=True)
        if self.create_msg:
            msg = 90 * "#" + "\n"
            msg += "\t\t\t\tPRECONDITIONED " if precond else "\t\t\t\t"
            msg += "LINEAR %s SOLVER log file end\n" % ("STEEPEST-DESCENT" if self.steepest else "CONJUGATE GRADIENT")
            msg += 90 * "#" + "\n"
            if verbose:
                print(msg.replace(" log file", ""))
            if self.logger:
                self.logger.addToLog(msg)
        # Clear restart object
        self.restart.clear_restart()

        return


