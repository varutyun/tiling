import sys
import numpy as np
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import colorsys


def load_pieces(filename):
    pieces = []
    with open(filename, 'r') as infile:
        n_pieces = int(infile.readline().rstrip())
        for piece_idndex in range(n_pieces):
            newline = infile.readline()
            while newline.startswith('#'):
                newline = infile.readline()
            y, x = [int(s) for s in newline.rstrip().split(' ')]
            piece = []
            for i in range(y):
                row = [int(s) for s in infile.readline().rstrip().split(' ')]
                piece.append(row)
            pieces.append(np.array(piece, dtype=np.int32))
    return pieces


def color_pieces(pieces):
    colored_pieces = [(i+1) * p for i, p in enumerate(pieces)]
    colors = [(0, 0, 0)]
    mult = (5**0.5 + 1) / 2
    h = np.random.random()
    color = colorsys.hsv_to_rgb(h, 0.5, 0.95)
    colors.append(color)
    while len(colors) <= len(pieces):
        h += mult
        h %= 1
        color = colorsys.hsv_to_rgb(h, 0.5, 0.95)
        colors.append(color)
    return colored_pieces, colors


def area(piece):
    return piece.astype(np.bool).sum()

def rotate(piece, change):
    change = change % 4
    if change == 0:
        return piece
    elif change == 1:
        return np.rot90(piece)
    elif change == 2:
        return np.rot90(piece, 2)
    elif change == 3:
        return np.rot90(piece, axes=(1, 0))

def test_piece_location(field, piece, row, column, place=False):
    if piece.shape[0] + row > field.shape[0]:
        return False
    if piece.shape[1] + column > field.shape[1]:
        return False
    field_slice = field[
        row: row + piece.shape[0],
        column: column + piece.shape[1]
    ]
    # print(field_slice * piece)
    result = ~np.any(field_slice * piece)
    if result and place:
        field_slice += piece
    return result

def greedy_place_piece(field, piece):
    for row in range(field.shape[0] - piece.shape[0] + 1):
        for column in range(field.shape[1] - piece.shape[1] + 1):
            if test_piece_location(field, piece, row, column, place=True):
                return True
    return False


class SimpleGA(object):
    
    def __init__(self, 
                 shape, 
                 pieces, 
                 pop_size,
                 n_offsprings,
                 p_mutation=0.03,
                 n_generations=1000,
                 crossover='cycle',
                 color_map=None,
                 picture_step=10,
                 survivors=2,
                 retirement_policy='keep'
                ):
        self.shape = shape
        self.pieces = pieces
        self.pop_size = pop_size
        self.population, self.fitnesses, self.matrices = self.gen_init_population(len(pieces), pop_size)
        self.n_offsprings = n_offsprings
        self.p_mutation = p_mutation
        self.n_generations = n_generations
        self.color_map = color_map
        self.picture_step = picture_step
        crossover_methods = {
            'order': self.order_crossover,
            'cycle': self.cycle_crossover
        }
        if crossover not in crossover_methods:
            raise ValueError("unsupported crossover method")
        fittest_selection = {
            'keep': self.select_fittest_with_merge,
            'quota': self.select_fittest_with_quota
        }
        if retirement_policy not in fittest_selection:
            raise ValueError("unknown retirement policy")
        self.select_fittest = fittest_selection[retirement_policy]
        self.crossover = crossover_methods[crossover]
        self.survivors = survivors
    
    def gen_init_population(self, n_pieces, pop_size):
        population = []
        fitnesses = []
        matrices = []
        for i in range(pop_size):
            individual = np.array([
                np.random.permutation(n_pieces),
                np.random.randint(0, 4, n_pieces)
            ])
            population.append(individual)
            fitness, field = self.greedy_placement(individual)
            fitnesses.append(fitness)
            matrices.append(field)
        return population, fitnesses, matrices
    
    def order_crossover(self, parent_1, parent_2):
        copy_begin, copy_end = np.random.choice(parent_1.shape[1], size=2, replace=False)
        if copy_end < copy_begin:
            copy_begin, copy_end = copy_end, copy_begin
        child_1 = -np.ones_like(parent_1)
        child_2 = -np.ones_like(parent_1)
        child_1[:, copy_begin:copy_end + 1] = parent_1[:, copy_begin:copy_end + 1]
        child_2[:, copy_begin:copy_end + 1] = parent_2[:, copy_begin:copy_end + 1]
        copied_indexes_1 = set(parent_1[0, copy_begin:copy_end + 1])
        copied_indexes_2 = set(parent_2[0, copy_begin:copy_end + 1])
        self._copy_remaining_genes(parent_2, child_1, copied_indexes_1, copy_begin, copy_end)
        self._copy_remaining_genes(parent_1, child_2, copied_indexes_2, copy_begin, copy_end)
        return child_1, child_2
        
    def _copy_remaining_genes(self, parent, child, used_indexes, copy_begin, copy_end):
        child_index = 0
        # print(copy_begin, copy_end)
        # print(used_indexes)
        for parent_index in range(parent.shape[1]):
            if copy_begin <= child_index <= copy_end:
                child_index += copy_end - copy_begin + 1
            if parent[0, parent_index] not in used_indexes:
                child[:, child_index] = parent[:, parent_index]
                child_index += 1
        return None
    
    def cycle_crossover(self, parent_1, parent_2):
        parent_1_inverse = np.argsort(parent_1[0])
        mask = -np.ones_like(parent_1[0])
        cycle_index = 0
        for index in range(parent_1.shape[1]):
            if mask[index] < 0:
                cycle_start = index
                next_index = parent_1_inverse[parent_2[0, cycle_start]]
                mask[next_index] = cycle_index
                while next_index != cycle_start:
                    next_index = parent_1_inverse[parent_2[0, next_index]]
                    mask[next_index] = cycle_index
                cycle_index += 1
        child_1 = -np.ones_like(parent_1)
        child_2 = -np.ones_like(parent_1)
        bool_mask = ~(mask % 2).astype(np.bool)
        child_1[:, bool_mask] = parent_1[:, bool_mask]
        child_1[:, ~bool_mask] = parent_2[:, ~bool_mask]
        child_2[:, bool_mask] = parent_2[:, bool_mask]
        child_2[:, ~bool_mask] = parent_1[:, ~bool_mask]
        return child_1, child_2
    
    def mutate(self, individual):
        # swaping two genes
        if np.random.random() < self.p_mutation:
            x, y = np.random.choice(individual.shape[1], size=2, replace=False)
            tmp = individual[:, x].copy()
            individual[:, x] = individual[:, y]
            individual[:, y] = tmp
        # randomly changing rotations
        else:
            mask = np.random.random(individual.shape[1]) < self.p_mutation
            individual[1, mask] = np.random.randint(0, 4, np.sum(mask))
        return individual
    
    def greedy_placement(self, individual):
        field = np.zeros(self.shape, dtype=np.int32)
        covered_area = 0
        for piece_index, rotation in individual.T:
            rotated_piece = rotate(self.pieces[piece_index], rotation)
            if not greedy_place_piece(field, rotated_piece):
                break
            else:
                covered_area += area(rotated_piece)
        return covered_area, field
    
    def select_parents(self, fitnesses=None):
        if fitnesses is None:
            fitnesses = self.fitnesses
        total_fitness = np.sum(fitnesses)
        parent_index_1, parent_index_2 = np.random.choice(
            len(self.population), 
            size=2, 
            replace=False, 
            p=fitnesses / total_fitness
        )
        return parent_index_1, parent_index_2
        
    def generate_offsprings(self):
        offsprings = []
        offsprings_fitness = []
        offspring_matrices = []
        while len(offsprings) < self.n_offsprings:
            parent_index_1, parent_index_2 = self.select_parents()
            children = self.crossover(
                self.population[parent_index_1],
                self.population[parent_index_2],
            )
            children = [self.mutate(child) for child in children]
            offsprings += children
            for child in children:
                fitness, matrix = self.greedy_placement(child)
                offsprings_fitness.append(fitness)
                offspring_matrices.append(matrix)
        return offsprings, offsprings_fitness, offspring_matrices
    
    def select_fittest_with_merge(self, offsprings, offsprings_fitness, offspring_matrices):
        self.population += offsprings
        self.fitnesses += offsprings_fitness
        self.matrices += offspring_matrices
        top_indexes = np.argsort(self.fitnesses)[::-1][:self.pop_size]
        self.population = [self.population[i] for i in top_indexes]
        self.fitnesses = [self.fitnesses[i] for i in top_indexes]
        self.matrices = [self.matrices[i] for i in top_indexes]
        self.best_index = 0
    
    def select_fittest_with_quota(self, offsprings, offsprings_fitness, offspring_matrices):
        old_top_indexes = np.argsort(self.fitnesses)[::-1][:self.survivors]
        new_top_indexes = np.argsort(offsprings_fitness)[::-1][:self.pop_size - self.survivors]
       
        self.population = [self.population[i] for i in old_top_indexes] +             [offsprings[i] for i in new_top_indexes]
        self.fitnesses = [self.fitnesses[i] for i in old_top_indexes] +             [offsprings_fitness[i] for i in new_top_indexes]
        self.matrices = [self.matrices[i] for i in old_top_indexes] +             [offspring_matrices[i] for i in new_top_indexes]
        self.best_index = np.argmax(self.fitnesses)
    
    def run(self, generations=None):
        if generations is None:
            generations = self.n_generations
        for generation in range(1, generations):
            timeit_start = timer()
            offsprings, offsprings_fitness, offspring_matrices = self.generate_offsprings()
            self.select_fittest(offsprings, offsprings_fitness, offspring_matrices)
            timeit_end = timer()
            print("generation %s created in %0.3f seconds" % (generation, timeit_end - timeit_start))
            print("best result: %s, pop_size: %s" % (self.fitnesses[self.best_index], len(self.population)))
            if (self.color_map is not None and 
                (generation % self.picture_step == 0 or generation + 1 == generations)):
                draw_matrix(self.matrices[self.best_index], self.color_map, self.shape, path='pics/generation_%s.png' % generation)
        

def draw_matrix(matrix, color_map, shape, path='pics/figure1.png'):
    fig1 = plt.figure(figsize=(20,10))
    ax1 = fig1.add_subplot(111, aspect='equal')
    for y, row in enumerate(matrix):
        for x, val in enumerate(row):
            ax1.add_patch(patches.Rectangle((x, y), 1, 1, color=color_map[val]))
    plt.xlim(0, shape[1])
    plt.ylim(0, shape[0])
    fig1.savefig(path)


if __name__ == '__main__':
    test_pieces = load_pieces('data/test_pieces.txt')
    total_area = np.sum([area(p) for p in test_pieces])
    print('total area of all pieces: %s' % total_area)

    colored_pieces, color_map = color_pieces(test_pieces)
    np.random.seed(1543)
    ga = SimpleGA(
        [6, 49],
        colored_pieces,
        pop_size=200,
        n_offsprings=200,
        n_generations=100,
        color_map=color_map,
        crossover='order',
        retirement_policy='keep'
    )
    ga.run()

