class NoveltyEvaluator:
    def __init__(self, dimension, k=30, pop_size=6):
        self.k = k  # sparseness value
        self.archive_threshold = 1.0 / dimension
        self.archive_threshold_min = 0.05 * self.archive_threshold
        self.archive_threshold_factor = 1.0
        self.archive_under = 0
        self.archive_under_threshold = 10
        self.archive_over_threshold = max(1, round(pop_size * 0.01))

        self.archive = []
        self.to_archive = []

    def reset(self):
        self.archive.clear()
        self.to_archive.clear()
        self.archive_under = 0

    def evaluate_novelty(self, current_pop, individual):
        total_size = len(self.archive) + len(current_pop)
        dist = []
        index = 0
        in_archive_count = 0

        for archived in self.archive:
            dist[index] = self._distance_between(individual, archived)

            if dist[index] < 0.0000001:
                in_archive_count += 1

            index += 1

        for other in current_pop:
            dist[index] = self._distance_between(individual, other)

            index += 1

        k_temp = min(total_size, self.k)
        dist.sort()
        avg_dist = 0
        for i in range(k_temp):
            avg_dist += dist[i]

        avg_dist /= k_temp

        if in_archive_count < self.k:
            if not self._contains_similar(current_pop, individual) \
                    and not self._contains_similar(self.to_archive, individual):
                self.to_archive.append(individual)

        return avg_dist

    def post_evaluation(self):
        if len(self.to_archive) == 0:
            self.archive_under += 1

            if self.archive_under == self.archive_under_threshold:
                self.archive_threshold /= self.archive_threshold_factor

                if self.archive_threshold < self.archive_threshold_min:
                    self.archive_threshold = self.archive_threshold_min

                self.archive_under = 0
        else:
            self.archive_under = 0

            if len(self.to_archive) > self.archive_over_threshold:
                self.archive_threshold *= self.archive_threshold_factor

        self.archive.extend(self.to_archive)
        self.to_archive.clear()

    def _distance_between(self, ind1, ind2):
        return 0

    def _contains_similar(self, population, individual):
        for ind in population:
            if self._distance_between(individual, ind) < self.archive_threshold:
                return True

        return False
