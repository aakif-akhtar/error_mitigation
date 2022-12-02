from gc import callbacks
from qiskit import IBMQ
from qiskit.providers.ibmq import least_busy

# from qiskit import QuantumCircuit, execute

from qiskit.providers.aer.noise import NoiseModel

from libraries.vqe_ansatz import vqe

import numpy as np

# from mitiq.zne.scaling import fold_gates_from_left, fold_gates_from_right, fold_global, fold_all
# from mitiq.zne.inference import LinearFactory, RichardsonFactory, PolyFactory
# from mitiq.zne.zne import execute_with_zne
# from mitiq.zne.zne import execute_with_zne

from qiskit.opflow import StateFn, CircuitStateFn

from qiskit import Aer
from qiskit.test.mock import FakeVigo
from qiskit.utils import QuantumInstance
from qiskit.opflow import CircuitSampler, StateFn, ExpectationFactory

# from qiskit.algorithms.optimizers import COBYLA, SLSQP, SPSA
from scipy.optimize import minimize


IBMQ.save_account(
    "4d7226ec37d38370454cbd7111e98a765d1cc1e6de0fa93a28a7b5918b6dd4b5c91f9fe74c748e270d8da2b5ef7c451b6550df0973de468ec4965a26b1b5c348",
    overwrite=True,
)
provider = IBMQ.load_account()
provider = IBMQ.get_provider("ibm-q")
# device = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= 3 and
#                                    not x.configuration().simulator and x.status().operational==True))
# print("Running on current least busy device: ", device)


# Build noise model from backend properties
# provider = IBMQ.load_account()
# backend = provider.get_backend(str(device))
backend = provider.get_backend("ibm_nairobi")

noise_model = NoiseModel.from_backend(backend)

# Get coupling map from backend
coupling_map = backend.configuration().coupling_map

# Get basis gates from noise model
basis_gates = noise_model.basis_gates


def executor(ansatz):
    # getting the hamiltonian for expectation value of energy eigenvalue
    qubit_op = vqe(ansatz_id=4).create_hamiltonian()

    # creating the wavefunction from the circuit
    psi = CircuitStateFn(ansatz)

    # define your backend or quantum instance (where you can add settings)
    # backend = provider.get_backend(str(device))
    backend = Aer.get_backend("qasm_simulator")
    # q_instance = QuantumInstance(backend, shots=1024)
    q_instance = QuantumInstance(
        backend, shots=1024, noise_model=noise_model, coupling_map=coupling_map
    )

    # define the state to sample

    # convert to expectation value
    expectation = ExpectationFactory.build(operator=qubit_op, backend=q_instance)
    measurable_expression = expectation.convert(StateFn(qubit_op, is_measurement=True))
    expect_op = measurable_expression.compose(psi).reduce()
    # get state sampler (you can also pass the backend directly)
    sampled_expect_op = CircuitSampler(q_instance).convert(expect_op)
    energy_evaluation = np.real(sampled_expect_op.eval())

    # evaluate
    # print('Sampled:', sampler.eval().real)

    return energy_evaluation

def objective_function_non_zne(params):

    ansatze = vqe(ansatz_id=4).get_ansatz(params)
    expectation = executor(ansatze)
    return expectation


if __name__ == "__main__":

    # energy_vals = []
    np.random.seed(0)
    # def vqe_callback(eval_count, opt_params, mean, stddev):
    #     print(eval_count, "mean=", mean)
    #     energy_vals.append(mean)
    def callbackF(Xi):
        global Nfeval
        print ('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}'.format(Nfeval, Xi[0], Xi[1], Xi[2], objective_function_non_zne(Xi)))
        Nfeval += 1

    for i in range(1):

        # optimizer = SPSA(maxiter=5, callback=vqe_callback)
        # optimizer = SPSA(maxiter=5)

        num_vars = 2 * vqe(ansatz_id=4).get_num_qubits() * vqe(ansatz_id=4).repitition
        params = np.random.randn(num_vars)
        # print('{0:4s}   {1:9s}   {2:9s}   {3:9s}   {4:9s}'.format('Iter', ' X1', ' X2', ' X3', 'f(X)'))
        non_mit_out = minimize(fun=objective_function_non_zne, x0=params, method='COBYLA', callback=callbackF)
        print("iteration %d non - mitigated results is : " % (i + 1), non_mit_out)