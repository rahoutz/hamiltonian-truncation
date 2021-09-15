# #################################################################################################################
# ######                                                                                        ####################
# ######        Hamiltonain Truncation                                                          ####################
# ######        Authors: Rachel Houtz, Kara Farnsworth, Markus Luty                             ####################
# ######                                                                                        ####################
# ######        This python code can be used to quickly generate the truncated Hamiltonian      ####################
# ######        for 2d \phi^4 theory. The interacting Hamiltonian matrix in a basis truncated   ####################
# ######        at ~10^6 states can be generated on O(days) using this code with modest         ####################
# ######        computational resources (for example, a laptop). See the accompanying           ####################
# ######        work arXiv:2109.XXXX.                                                           ####################
# ######                                                                                        ####################
# ##################################################################################################################
import math
import numpy as np
import scipy
from scipy import sparse
from scipy.stats import uniform
from scipy.sparse import csr_matrix
from scipy.sparse import identity
import scipy.sparse.linalg as sparse_la
import time
import os.path
import bisect

# #
# # ##################################################################################################################
# # ################# GLOBAL VARIABLES ############################################################################
# # ##################################################################################################################
PATH = 'data_files/'
dir_name = "{}".format(PATH)
if not os.path.isdir(dir_name): os.mkdir(dir_name)
basis_dir_name = "{}".format(PATH) + "BasisData/"
if not os.path.isdir(basis_dir_name): os.mkdir(basis_dir_name)
omega_list = []


# # ##################################################################################################################
# # ################# GENERATE THE BASIS OF STATES ###################################################################
# # ##################################################################################################################

def omega(l, m, r):
    output = math.sqrt(float(l) * float(l) / (r*r) + m*m)
    return output


def gen_omega_list(lmax, m, r):
    global omega_list
    omega_list = []
    for l in range(lmax+1):
        omega_list.append(omega(l, m, r))


def make_basis(lmax, e_max, m, r):
    global omega_list
    basis = gen_basis(lmax, e_max, m, r)
    gen_omega_list(lmax, m, r)
    return basis


def gen_basis(lmax, e_max, m, r):
    result = []
    if lmax == 0:
        result = list(map(lambda x: [x], list(range(0, int(math.floor(e_max/m)) + 1))))
    else:
        for n in range(0, int(math.floor(e_max/omega(lmax, m, r))) + 1):
            prev_list = gen_basis(lmax-1, e_max-n * omega(lmax, m, r), m, r)
            for NP in range(0, n+1):
                result = result + list(map(lambda x: list(x) + [NP, n-NP], prev_list))
    return result



def ix(l, sigma):
    if l == 0:
        return 0
    elif sigma >= 0:
        return 2*l-1
    else:
        return 2*l


def the_l(ix_arg):
    if ix_arg == 0:
        return 0
    elif ix_arg % 2 != 0:
        return int((ix_arg+1) / 2)
    else:
        return int(ix_arg/2)


def the_sigma(ix_arg):
    if ix_arg == 0:
        return 0
    elif ix_arg % 2 != 0:
        return 1
    else:
        return -1


def count_particles(state):
    num_particles = 0
    for particles in state:
        num_particles = num_particles+particles
    return num_particles


def l_total(state):
    l_sum = 0
    for i in range(1, len(state)):
        if i % 2 != 0:
            l_sum = l_sum + state[i] * the_l(i)
        else:
            l_sum = l_sum-state[i] * the_l(i)
    return l_sum


def state_energy(state, m, r):
    global omega_list
    total_energy = 0
    for i in range(len(state)):
        total_energy = total_energy+state[i] * omega_list[abs(the_l(i))]
    return total_energy


def find_pos(state, basis, min_index, max_index):
    index = bisect.bisect_left(basis, state, min_index, max_index)
    if basis[index] == state:
        return index
    else:
        return -1



def basis_odd(basis):
    new_basis = []
    for state in basis:
        if sum(state) % 2 != 0:
            new_basis.append(state)
    return new_basis


def basis_even(basis):
    new_basis = []
    for state in basis:
        if sum(state) % 2 == 0:
            new_basis.append(state)
    return new_basis


def basis_l0(basis):
    new_basis = []
    for state in basis:
        if l_total(state) == 0:
            new_basis.append(state)
    return new_basis



def basis_nmax(basis, n_max):
    new_basis = []
    for state in basis:
        if count_particles(state) <= n_max:
            new_basis.append(state)
    return new_basis
#
#
# ##################################################################################################################
# ###### RAISING AND LOWERING OPERATORS ############################################################################
# ##################################################################################################################
#
#
#
# ###### A SINGLE RAISING/LOWERING OPERATOR ########################################################################
# Note used in final version but demonstrative of combined operators
def raising(l, sigma, basis):
    row = []
    col = []
    data = []
    length_basis = len(basis)
    for state_index in range(length_basis):
        pos = []
        state=basis[state_index]
        if ix(l, sigma) <= len(state):
            newstate = [state[ix(l, sigma)]+1 if i == ix(l, sigma) else state[i] for i in range(len(state))]
            if newstate in basis:
                pos = basis.index(newstate)
        if pos != -1:
            row.append(state_index)
            col.append(pos)
            data.append(math.sqrt(state[ix(l, sigma)]+1))
    return csr_matrix((data, (row, col)), shape=(length_basis, length_basis))


# Note used in final version but demonstrative of combined operators
def lowering(l, sigma, basis):
    row = []
    col = []
    data = []
    length_basis = len(basis)
    for state_index in range(length_basis):
        pos = []
        state=basis[state_index]
        if ix(l, sigma) <= len(state):
            newstate = [state[ix(l, sigma)]-1 if i == ix(l, sigma) else state[i] for i in range(len(state))]
            if newstate in basis:
                pos = basis.index(newstate)
        if pos != -1:
            row.append(state_index)
            col.append(pos)
            data.append(math.sqrt(state[ix(l, sigma)]))
    return csr_matrix((data, (row, col)), shape=(length_basis, length_basis))

####### A SET OF RAISING AND LOWERING OPERATORS (with given l's) ########################################################################

def lower4(l1, s1, l2, s2, l3, s3, l4, s4, e_max, m, r, basis):
    row = []
    col = []
    data = []
    norm = 0
    ix_max = len(basis[0])
    length_basis = len(basis)
    for state_index in range(length_basis):
        pos = -1
        state=basis[state_index]
        norm = math.sqrt(state[ix(l1, s1)])
        if norm != 0:
            newstate = [state[ix(l1, s1)] - 1 if i == ix(l1, s1) else state[i] for i in range(ix_max)]
            norm = norm*math.sqrt(newstate[ix(l2, s2)])
            if norm != 0:
                newstate2 = [newstate[ix(l2, s2)] - 1 if i == ix(l2, s2) else newstate[i] for i in range(ix_max)]
                norm = norm*math.sqrt(newstate2[ix(l3, s3)])
                if norm != 0:
                    newstate3 = [newstate2[ix(l3, s3)] - 1 if i == ix(l3, s3) else newstate2[i] for i in range(ix_max)]
                    norm = norm*math.sqrt(newstate3[ix(l4, s4)])
                    if norm != 0:
                        newstate4 = [newstate3[ix(l4, s4)] - 1 if i == ix(l4, s4) else newstate3[i] for i in range(ix_max)]
                        if state_energy(newstate4, m, r) <= e_max:
                            pos = bisect.bisect_left(basis, newstate4)
        if pos != -1:
            row.append(state_index)
            col.append(pos)
            data.append(norm)
    return csr_matrix((data, (row, col)), shape=(length_basis, length_basis))


def raise1low3(l1, s1, l2, s2, l3, s3, l4, s4, e_max, m, r, basis):
    # l4, s4 is the raising operator
    row = []
    col = []
    data = []
    norm = 0
    ix_max = len(basis[0])
    length_basis = len(basis)
    for state_index in range(length_basis):
        pos = -1
        state = basis[state_index]
        norm = math.sqrt(state[ix(l1, s1)])
        if norm != 0:
            newstate = [state[ix(l1, s1)] - 1 if i == ix(l1, s1) else state[i] for i in range(ix_max)]
            norm = norm*math.sqrt(newstate[ix(l2, s2)])
            if norm != 0:
                newstate2 = [newstate[ix(l2, s2)] - 1 if i == ix(l2, s2) else newstate[i] for i in range(ix_max)]
                norm = norm*math.sqrt(newstate2[ix(l3, s3)])
                if norm != 0:
                    newstate3 = [newstate2[ix(l3, s3)] - 1 if i == ix(l3, s3) else newstate2[i] for i in range(ix_max)]
                    norm = norm*math.sqrt(newstate3[ix(l4, s4)] + 1)
                    if norm != 0:
                        newstate4 = [newstate3[ix(l4, s4)] + 1 if i == ix(l4, s4) else newstate3[i] for i in range(ix_max)]
                        if state_energy(newstate4, m, r) <= e_max:
                            pos = bisect.bisect_left(basis, newstate4)
        if pos != -1:
            row.append(state_index)
            col.append(pos)
            data.append(norm)
    return csr_matrix((data, (row, col)), shape=(length_basis, length_basis))


def raise2low2(l1, s1, l2, s2, l3, s3, l4, s4, e_max, m, r, basis):
    row = []
    col = []
    data = []
    norm = 0
    ix_max = len(basis[0])
    length_basis = len(basis)
    for state_index in range(length_basis):
        state=basis[state_index]
        pos = -1
        norm = math.sqrt(state[ix(l1, s1)])
        if norm != 0:
            newstate = [state[ix(l1, s1)] - 1 if i == ix(l1, s1) else state[i] for i in range(ix_max)]
            norm = norm*math.sqrt(newstate[ix(l2, s2)])
            if norm != 0:
                newstate2 = [newstate[ix(l2, s2)] - 1 if i == ix(l2, s2) else newstate[i] for i in range(ix_max)]
                norm = norm*math.sqrt(newstate2[ix(l3, s3)]+1)
                if norm != 0:
                    newstate3 = [newstate2[ix(l3, s3)] + 1 if i == ix(l3, s3) else newstate2[i] for i in range(ix_max)]
                    norm = norm*math.sqrt(newstate3[ix(l4, s4)] + 1)
                    if norm != 0:
                        newstate4 = [newstate3[ix(l4, s4)] + 1 if i == ix(l4, s4) else newstate3[i] for i in range(ix_max)]
                        if newstate4==state:
                            pos = state_index
                        elif state_energy(newstate4, m, r) <= e_max:
                            pos = bisect.bisect_left(basis, newstate4)
        if pos != -1:
            row.append(state_index)
            col.append(pos)
            data.append(norm)
    return csr_matrix((data, (row, col)), shape=(length_basis, length_basis))


def lower2(l1, s1, l2, s2, e_max, m, r, basis):
    row = []
    col = []
    data = []
    norm = 0
    ix_max = len(basis[0])
    length_basis = len(basis)
    for state_index in range(length_basis):
        pos = -1
        state = basis[state_index]
        norm = math.sqrt(state[ix(l1, s1)])
        if norm != 0:
            newstate = [state[ix(l1, s1)] - 1 if i == ix(l1, s1) else state[i] for i in range(ix_max)]
            norm = norm*math.sqrt(newstate[ix(l2, s2)])
            if norm != 0:
                newstate2 = [newstate[ix(l2, s2)] - 1 if i == ix(l2, s2) else newstate[i] for i in range(ix_max)]
                if state_energy(newstate2, m, r) <= e_max:
                    pos = bisect.bisect_left(basis, newstate2)
        if pos != -1:
            row.append(state_index)
            col.append(pos)
            data.append(norm)
    return csr_matrix((data, (row, col)), shape=(length_basis, length_basis))


def raise1low1(l1, s1, l2, s2, e_max, m, r, basis):
    row = []
    col = []
    data = []
    norm = 0
    ix_max = len(basis[0])
    length_basis=len(basis)
    for state_index in range(length_basis):
        pos = -1
        state=basis[state_index]
        norm = math.sqrt(state[ix(l1, s1)])
        if norm != 0:
            newstate = [state[ix(l1, s1)] - 1 if i == ix(l1, s1) else state[i] for i in range(ix_max)]
            norm = norm*math.sqrt(newstate[ix(l2, s2)] + 1)
            if norm != 0:
                newstate2 = [newstate[ix(l2, s2)] + 1 if i == ix(l2, s2) else newstate[i] for i in range(ix_max)]
                if newstate2==state:
                    pos=state_index
                elif state_energy(newstate2, m, r) <= e_max:
                    pos = bisect.bisect_left(basis, newstate2)
        if pos != -1:
            row.append(state_index)
            col.append(pos)
            data.append(norm)
    return csr_matrix((data, (row, col)), shape=(length_basis, length_basis))


#
#
#
#
# ################ SIMPLIFY AND GATHER TERMS IN THE SUMS OF OPERATORS ###############################
#


def raise4_op_list(lmax, e_max, m, r):
    global omega_list
    op_list = []
# -lmax <= j < k < l < n <= lmax
    for j in range(-lmax, lmax-2):
        for k in range(j+1, lmax-1):
            for l in range(k+1, lmax):
                n = -l-k-j
                if n in range(l+1, lmax+1) and omega_list[abs(j)] + omega_list[abs(k)] +omega_list[abs(l)] + omega_list[abs(n)] <= e_max:
                    op = [j, k, l, n]
                    op_list.append([op, 24])
# -lmax <= j < k < l = n <= lmax
    for j in range(-lmax, lmax-1):
        for k in range(j+1, lmax):
            if (-k-j) % 2 == 0:
                l = int((-k-j)/2)
                if l in range(k+1, lmax+1) and omega_list[abs(j)] + omega_list[abs(k)] + 2 * omega_list[abs(l)] <= e_max:
                    op = [j, k, l, l]
                    op_list.append([op, 12])
# -lmax <= j < k = n < l <= lmax
    for j in range(-lmax, lmax-1):
        for k in range(j+1, lmax):
            l = -2 * k-j
            if l in range(k+1, lmax+1) and omega_list[abs(j)] + 2 * omega_list[abs(k)] + omega_list[abs(l)] <= e_max:
                op = [j, k, k, l]
                op_list.append([op, 12])
# -lmax <= j = n < k < l <= lmax
    for j in range(-lmax, lmax-1):
        for k in range(j+1, lmax):
            l = -k-2 * j
            if l in range(k+1, lmax+1) and 2 * omega_list[abs(j)] + omega_list[abs(k)] + omega_list[abs(l)] <= e_max:
                op = [j, j, k, l]
                op_list.append([op, 12])
# -lmax <= j = k < l= n <= lmax
    for k in range(-lmax, lmax):
        l = -k
        if l in range(k+1, lmax+1) and 2 * omega_list[abs(k)] + 2 * omega_list[abs(l)] <= e_max:
            op = [k, k, l, l]
            op_list.append([op, 6])
# -lmax <= j < k = l = n <= lmax
    for j in range(-lmax, lmax):
        if j % 3 == 0:
            l = - j / 3
            l = int(l)
            if l in range(j+1, lmax+1) and omega_list[abs(j)] + 3 * omega_list[abs(l)] <= e_max:
                op = [j, l, l, l]
                op_list.append([op, 4])
# -lmax <= j  = l = n < k <= lmax
    for j in range(-lmax, lmax):
        k = -3 * j
        if k in range(j+1, lmax+1) and 3 * omega_list[abs(j)] + omega_list[abs(k)] <= e_max:
            op = [j, j, j, k]
            op_list.append([op, 4])
# -lmax <= j = k = l = n <= lmax
    op = [0, 0, 0, 0]
    op_list.append([op, 1])
    return op_list



def raise31_op_list(lmax, e_max, m, r):
    global omega_list
    op_list = []
    # -lmax <= j < k < l <= lmax
    for j in range(-lmax, lmax-1):
        for k in range(j+1, lmax):
            for l in range(k+1, lmax+1):
                n = l + k + j
                if n in range(-lmax, lmax+1) and omega_list[abs(j)] + omega_list[abs(k)] +omega_list[abs(l)] <= e_max and omega_list[abs(n)] <= e_max:
                    op = [j, k, l, n]
                    op_list.append([op, 6])
    # -lmax <= j < k = l <= lmax
    for j in range(-lmax, lmax):
        for k in range(j+1, lmax+1):
            n = j + 2 * k
            if n in range(-lmax, lmax+1) and omega_list[abs(j)] + 2 * omega_list[abs(k)] <= e_max and omega_list[abs(n)] <= e_max:
                op = [j, k, k, n]
                op_list.append([op, 3])
    # -lmax <= j = k < l <= lmax
    for k in range(-lmax, lmax):
        for l in range(k+1, lmax+1):
            n = 2 * k + l
            if n in range(-lmax, lmax+1) and 2 * omega_list[abs(k)] + omega_list[abs(l)] <= e_max and omega_list[abs(n)] <= e_max:
                op = [k, k, l, n]
                op_list.append([op, 3])
    # -lmax <= j = k = l <= lmax
    for k in range(math.ceil(-lmax/3), math.floor(lmax/3) + 1):
        n = 3 * k
        if n in range(-lmax, lmax+1) and 3 * omega_list[abs(k)] <= e_max and omega_list[abs(n)] <= e_max:
            op = [k, k, k, n]
            op_list.append([op, 1])
    return op_list


def raise22_op_list(lmax, e_max, m, r):
    global omega_list
    op_list = []
    # -lmax <= j < k <= lmax
    # -lmax <= l < n <= lmax
    for j in range(-lmax, lmax):
        for k in range(j+1, lmax+1):
            for l in range(-lmax, lmax):
                n = -l + k + j
                if n in range(l+1, lmax+1) and omega_list[abs(j)] + omega_list[abs(k)] <= e_max and omega_list[abs(l)] + omega_list[abs(n)] <= e_max:
                    op = [j, k, l, n]
                    op_list.append([op, 4])
    # -lmax <= j < k <= lmax
    # -lmax <= l = n <= lmax
    for j in range(-lmax, lmax):
        for k in range(j+1, lmax+1):
            if (k+j) % 2 == 0:
                l = int((k+j)/2)
                if l in range(-lmax, lmax+1) and omega_list[abs(j)] + omega_list[abs(k)] <= e_max and 2 * omega_list[abs(l)] <= e_max:
                    op = [j, k, l, l]
                    op_list.append([op, 2])
    # -lmax <= j = k <= lmax
    # -lmax <= l < n <= lmax
    for l in range(-lmax, lmax):
        for n in range(l+1, lmax+1):
            if (l+n) % 2 == 0:
                k = int((l+n)/2)
                if k in range(-lmax, lmax+1) and 2 * omega_list[abs(k)] <= e_max and omega_list[abs(l)] + omega_list[abs(n)] <= e_max:
                    op = [k, k, l, n]
                    op_list.append([op, 2])
    # -lmax <= j = k <= lmax
    # -lmax <= l = n <= lmax
    for k in range(-lmax, lmax+1):
        l = k
        if 2 * omega_list[abs(k)] <= e_max and 2 * omega_list[abs(l)] <= e_max:
            op = [k, k, l, l]
            op_list.append([op, 1])
    return op_list


def raise2_op_list(lmax, e_max, m, r):
    global omega_list
    op_list = []
    # -lmax <= i < j <= lmax
    for i in range(-lmax, 0):
        if 2 * omega_list[abs(i)] <= e_max:
            op = [i, -i]
            op_list.append([op, 2])
    op = [0, 0]
    op_list.append([op, 1])
    return op_list


def raise11_op_list(lmax):
    op_list = []
    # -lmax <= i <= lmax
    for i in range(-lmax, lmax+1):
        op = [i, i]
        op_list.append([op, 1])
    return op_list
#
#
#
# ##################################################################################################################
# ################ H MATRICES AND TERMS  ############################################################################
# ###################################################################################################################
#

def term_4low(lmax, e_max, m, r, basis):
    global omega_list
    length_basis=len(basis)
    total_op = csr_matrix(([], ([], [])), shape=(length_basis, length_basis))
    op_list = raise4_op_list(lmax, e_max, m, r)
    for op in op_list:
        i = op[0][0]
        j = op[0][1]
        l = op[0][2]
        k = op[0][3]
        new_op = lower4(int(abs(k)), np.sign(k), int(abs(l)), np.sign(l), int(abs(j)), np.sign(j), int(abs(i)), np.sign(i), e_max, m, r, basis)
        if new_op.count_nonzero() != 0:
            factor = op[1]/math.sqrt(omega_list[abs(i)] * omega_list[abs(j)] * omega_list[abs(l)] * omega_list[abs(k)])
            total_op = total_op+factor*new_op
    return total_op


def term_1raise3low(lmax, e_max, m, r, basis):
    global omega_list
    length_basis = len(basis)
    total_op = csr_matrix(([], ([], [])), shape=(length_basis, length_basis))
    op_list = raise31_op_list(lmax, e_max, m, r)
    for op in op_list:
        i = op[0][0]
        j = op[0][1]
        l = op[0][2]
        # k is the raising operator
        k = op[0][3]
        new_op = raise1low3(int(abs(i)), np.sign(i), int(abs(j)), np.sign(j), int(abs(l)), np.sign(l), int(abs(k)), np.sign(k), e_max, m, r, basis)
        if new_op.count_nonzero() != 0:
            factor = 4 * op[1] / math.sqrt(omega_list[abs(i)] * omega_list[abs(j)] * omega_list[abs(l)] * omega_list[abs(k)])
            total_op = total_op + factor * new_op
    return total_op


def term_2raise2low(lmax, e_max, m, r, basis):
    global omega_list
    length_basis=len(basis)
    total_op = csr_matrix(([], ([], [])), shape=(length_basis, length_basis))
    op_list = raise22_op_list(lmax, e_max, m, r)
    for op in op_list:
        i = op[0][0]
        j = op[0][1]
        l = op[0][2]
        k = op[0][3]
        new_op = raise2low2(abs(k), np.sign(k), abs(l), np.sign(l), abs(j), np.sign(j), abs(i), np.sign(i), e_max, m, r, basis)
        if new_op.count_nonzero() != 0:
            factor = 6 * op[1] / math.sqrt(omega_list[abs(i)] * omega_list[abs(j)] * omega_list[abs(l)] * omega_list[abs(k)])
            total_op = total_op + factor * new_op
    return total_op


def term_2low(lmax, e_max, m, r, basis):
    global omega_list
    length_basis = len(basis)
    total_op = csr_matrix(([], ([], [])), shape=(length_basis, length_basis))
    op_list = raise2_op_list(lmax, e_max, m, r)
    for op in op_list:
        i = op[0][0]
        j = op[0][1]
        new_op = lower2(abs(j), np.sign(j), abs(i), np.sign(i), e_max, m, r, basis)
        if new_op.count_nonzero() != 0:
            factor = op[1]/math.sqrt(omega_list[abs(i)] * omega_list[abs(j)])
            total_op = total_op+factor*new_op
    return total_op


def term_1raise1low(lmax, e_max, m, r, basis):
    global omega_list
    row = []
    col = []
    data = []
    length_basis=len(basis)
    for state_index in range(length_basis):
        state = basis[state_index]
        row.append(state_index)
        col.append(state_index)
        factor = 0
        for i in range(len(state)):
            factor = factor + 2 * state[i] / omega_list[abs(the_l(i))]
        data.append(factor)
    return csr_matrix((data, (row, col)), shape=(length_basis, length_basis))




###################################################################################################################
###################################################################################################################
################# THE H MATRIX ONCE THE BASIS IS BUILT #############################################################
###################################################################################################################
###################################################################################################################

def delta_h4(lmax, e_max, m, r, basis):
    term4 = term_4low(lmax, e_max, m, r, basis)
    term3 = term_1raise3low(lmax, e_max, m, r, basis)
    h4mat = term4 + term4.transpose() + term_2raise2low(lmax, e_max, m, r, basis) + term3 + term3.transpose()
    return h4mat



def delta_h2(lmax, e_max, m, r, basis):
    term2 = term_2low(lmax, e_max, m, r, basis)
    return term2 + term2.transpose() + term_1raise1low(lmax, e_max, m, r, basis)






def h0(lmax, e_max, m, r, basis):
    # start=time.time()
    length_basis = len(basis)
    row = []
    col = []
    data = []
    for state_index in range(length_basis):
        state = basis[state_index]
        row.append(state_index)
        col.append(state_index)
        data.append(state_energy(state, m, r))
    return csr_matrix((data, (row, col)), shape=(length_basis, length_basis))




def h_full(lmax, e_max, m, delta_m, basis, g, r):
    if pow(m, 2) < pow(e_max, 2) / 4:
        lmaxeff = int(min(lmax, math.floor(math.sqrt(pow(r, 2) * (float(pow(e_max,2))/4. - pow(m, 2))))))
    else:
        lmaxeff = 0
    return h0(lmaxeff, e_max, m, r, basis) + 1./4.*delta_m*delta_h2(lmaxeff, e_max, m, r, basis) +g/r*1/(8*math.pi) *1/24.*delta_h4(lmax, e_max, m, r, basis)
#
#


###################################################################################################################
###################################################################################################################
################# GENERATE H MATRIX  ##############################################################################
###################################################################################################################
###################################################################################################################


def gen_h_mats(lmax, e_max, m, r):
    if pow(m, 2) < pow(e_max, 2) / 4:
        lmaxeff = int(min(lmax, math.floor(math.sqrt(pow(r, 2) * (float(pow(e_max, 2)) / 4. - pow(m, 2))))))
    else:
        lmaxeff = 0
    dir_name = "{}".format(PATH) + "BasisData/m{:.2f}".format(m) + "_r" + "{:.2f}".format(r) + "/"
    if not os.path.isdir(dir_name): os.mkdir(dir_name)
    H0file = dir_name + "l{}".format(lmaxeff) + "_E{:.2f}_H0.npz".format(e_max)
    H2file = dir_name + "l{}".format(lmaxeff) + "_E{:.2f}_H2.npz".format(e_max)
    H4file = dir_name + "l{}".format(lmaxeff) + "_E{:.2f}_H4.npz".format(e_max)
    ################################################################################
    #   These lines allow you to not regenerate matrices you've already made (time saver):
    #
    #    if not (os.path.isfile(H0file) and os.path.isfile(H2file) and os.path.isfile(H4file)):
    #        basis = basis_l0(make_basis(lmaxeff, e_max, m, r))
    #        basis.sort()
    #    else: basis = []
    #    if not os.path.isfile(H0file):
    #        scipy.sparse.save_npz(H0file, h0(lmaxeff, e_max, m, r, basis))
    #    if not os.path.isfile(H2file):
    #        scipy.sparse.save_npz(H2file, delta_h2(lmaxeff, e_max, m, r, basis))
    #    if not os.path.isfile(H4file):
    #        scipy.sparse.save_npz(H4file, delta_h4(lmaxeff, e_max, m, r, basis))
    ###############################################################################
    basis = basis_l0(make_basis(lmaxeff, e_max, m, r))
    basis.sort()
    scipy.sparse.save_npz(H0file, h0(lmaxeff, e_max, m, r, basis))
    scipy.sparse.save_npz(H2file, delta_h2(lmaxeff, e_max, m, r, basis))
    scipy.sparse.save_npz(H4file, delta_h4(lmaxeff, e_max, m, r, basis))

###### GENERATE THE MATRIX ONLY USING Z2 EVEN STATES IN THE BASIS (One could ammend to to only use Odd states, stop restricting to total l = 0, etc)

def gen_h_mats_even(lmax, e_max, m, r):
    if pow(m, 2) < pow(e_max, 2) / 4:
        lmaxeff = int(min(lmax, math.floor(math.sqrt(pow(r, 2) * (float(pow(e_max, 2)) / 4. - pow(m, 2))))))
    else:
        lmaxeff = 0
    dir_name = "{}".format(PATH) + "BasisData/m{:.2f}".format(m) + "_r" + "{:.2f}".format(r) + "_even/"
    if not os.path.isdir(dir_name): os.mkdir(dir_name)
    H0file = dir_name + "l{}".format(lmaxeff) + "_E{:.2f}_H0.npz".format(e_max)
    H2file = dir_name + "l{}".format(lmaxeff) + "_E{:.2f}_H2.npz".format(e_max)
    H4file = dir_name + "l{}".format(lmaxeff) + "_E{:.2f}_H4.npz".format(e_max)
################################################################################
#   These lines allow you to not regenerate matrices you've already made (time saver):
#
#    if not (os.path.isfile(H0file) and os.path.isfile(H2file) and os.path.isfile(H4file)):
#        basis = basis_even(basis_l0(make_basis(lmaxeff, e_max, m, r)))
#        basis.sort()
#    else: basis = []
#    if not os.path.isfile(H0file):
#        scipy.sparse.save_npz(H0file, h0(lmaxeff, e_max, m, r, basis))
#    if not os.path.isfile(H2file):
#        scipy.sparse.save_npz(H2file, delta_h2(lmaxeff, e_max, m, r, basis))
#    if not os.path.isfile(H4file):
#        scipy.sparse.save_npz(H4file, delta_h4(lmaxeff, e_max, m, r, basis))
###############################################################################
    basis = basis_even(basis_l0(make_basis(lmaxeff, e_max, m, r)))
    basis.sort()
    scipy.sparse.save_npz(H0file, h0(lmaxeff, e_max, m, r, basis))
    scipy.sparse.save_npz(H2file, delta_h2(lmaxeff, e_max, m, r, basis))
    scipy.sparse.save_npz(H4file, delta_h4(lmaxeff, e_max, m, r, basis))
##



###################################################################################################################
###################################################################################################################
################# DIAGONALIZE THE H MATRIX  ##############################################################################
###################################################################################################################
###################################################################################################################


def gen_eigens(lmax, e_max, m, delta_m, g, delta_g, r,  n_eigens, basis_name):
    output_array=[]
    if pow(m, 2) < pow(e_max, 2) / 4:
        lmaxeff = int(min(lmax, math.floor(math.sqrt(pow(r, 2) * (float(pow(e_max, 2)) / 4. - pow(m, 2))))))
    else:
        lmaxeff = 0
    matrix_dir_name = "{}".format(PATH) + "BasisData/m{:.2f}".format(m) + "_r" + "{:.2f}".format(r) + basis_name + "/"
    if os.path.isdir(matrix_dir_name):
        H0file = matrix_dir_name + "l{}".format(lmaxeff) + "_E{:.2f}_H0.npz".format(e_max)
        H2file = matrix_dir_name + "l{}".format(lmaxeff) + "_E{:.2f}_H2.npz".format(e_max)
        H4file = matrix_dir_name + "l{}".format(lmaxeff) + "_E{:.2f}_H4.npz".format(e_max)
        if os.path.isfile(H0file) and os.path.isfile(H2file) and os.path.isfile(H4file):
            h0mat = scipy.sparse.load_npz(H0file)
            length_basis = h0mat.shape[0]
            if n_eigens > length_basis: n_eigens = length_basis
            h2mat = scipy.sparse.load_npz(H2file)
            h4mat = scipy.sparse.load_npz(H4file)
            h_total = h0mat + 1. / 4. * delta_m * h2mat + (g + delta_g) / r * 1 / (8 * math.pi) * 1 / 24. * h4mat
            if n_eigens > length_basis-2:
                eigens = scipy.linalg.eigvals((h_total).toarray())
                output_array = np.asarray([lmaxeff, e_max])
                output_array = np.append(output_array, sorted(eigens.real)[:n_eigens])
            else:
                hs_norm=scipy.sparse.linalg.norm(h_total)
                eigens = sparse_la.eigs(h_total - hs_norm * identity(length_basis), n_eigens, None, None, which='LM',)[0]
                eigens = np.array(eigens.real) + hs_norm
                output_array = np.asarray([lmaxeff, e_max])
                output_array= np.append(output_array, sorted(eigens.real))
    return output_array

def gen_mat_eigens(lmax, e_max, m, delta_m, g, delta_g, r, n_eigens):
    output_array = []
    if pow(m, 2) < pow(e_max, 2) / 4:
        lmaxeff = int(min(lmax, math.floor(math.sqrt(pow(r, 2) * (float(pow(e_max, 2)) / 4. - pow(m, 2))))))
    else:
        lmaxeff = 0
    basis = basis_l0(make_basis(lmaxeff, e_max, m, r))
    basis.sort()
    h0mat = h0(lmaxeff, e_max, m, r, basis)
    length_basis = h0mat.shape[0]
    if n_eigens > length_basis: n_eigens = length_basis
    h2mat = delta_h2(lmaxeff, e_max, m, r, basis)
    h4mat = delta_h4(lmaxeff, e_max, m, r, basis)
    h_total = h0mat + 1. / 4. * delta_m * h2mat + (g + delta_g) / r * 1 / (8 * math.pi) * 1 / 24. * h4mat
    if n_eigens > length_basis-2:
        eigens = scipy.linalg.eigvals(h_total.toarray())
        output_array = np.asarray([lmaxeff, e_max])
        output_array = np.append(output_array, sorted(eigens.real)[:n_eigens])
    else:
        hs_norm=scipy.sparse.linalg.norm(h_total)
        eigens = sparse_la.eigs(h_total - hs_norm * identity(length_basis), n_eigens, None, None, which='LM',)[0]
        eigens = np.array(eigens.real) + hs_norm
        output_array = np.asarray([lmaxeff, e_max])
        output_array = np.append(output_array, sorted(eigens.real))
    return output_array


###################################################################################################################
###################################################################################################################
################# EXAMPLE CODE  ##############################################################################
###################################################################################################################
###################################################################################################################

M = 1
R=10./(2* math.pi)

for Emax in range(1,10):
    if pow(M, 2) < pow(Emax, 2) / 4:
        Lmax = int(math.floor(math.sqrt(pow(R, 2) * (float(pow(Emax, 2)) / 4. - pow(M, 2)))))
    else:
        Lmax = 0
    gen_h_mats(Lmax, Emax, M, R)


Lambda = 4*math.pi
Basis_name = ""
N_eigens = 25
for Emax in range(1,10):
    if pow(M, 2) < pow(Emax, 2) / 4:
        Lmax = int(math.floor(math.sqrt(pow(R, 2) * (float(pow(Emax, 2)) / 4. - pow(M, 2)))))
    else:
        Lmax = 0
    delta_g = 0 # possible counterterm
    delta_m = 0 # possible counterterm
    print(gen_eigens(Lmax, Emax, M, delta_m, Lambda, delta_g, R,  N_eigens, Basis_name))