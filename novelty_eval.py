import numpy as np

class NoveltyEvaluator:
    def __init__(self, pop_size, k):
        self.pop_size = pop_size
        self.k = k  # sparseness value
        self.archive_threshold = 0.00000001
        self.archive_threshold_min = 0.05 * self.archive_threshold
        self.archive_threshold_factor = 10
        self.archive_under = 0
        self.archive_under_threshold = 10
        self.archive_over_threshold = max(1, round(self.pop_size * 0.01))

        self.archive = []
        self.to_archive = []

    def reset(self):
        self.archive.clear()
        self.to_archive.clear()
        self.archive_threshold = 0.00000001
        self.archive_under = 0

    def evaluate_novelty(self, current_pop, predictions, behavior):
        total_size = len(self.archive) + len(current_pop)
        distances = []
        in_archive_count = 0

        for archived in self.archive:
            distance = self._distance_between(behavior, archived)
            if distance < 0.000000000001:
                in_archive_count += 1

            distances.append(distance)

        for other in current_pop:
            distances.append(self._distance_between(behavior, predictions.get(tuple(other))))

        k_temp = min(total_size - 1, self.k)
        distances.sort()
        avg_dist = 0
        for i in range(1, k_temp + 1):
            avg_dist += distances[i]

        avg_dist /= k_temp

        if in_archive_count < self.k:
            in_arch = self._contains_similar(self.archive, behavior)
            in_to_arch = self._contains_similar(self.to_archive, behavior)
            #print("in_arch=", in_arch, "in_to_arch=", in_to_arch)
            if not in_arch and not in_to_arch:
                #print("ADDING TO ARCHIVE!")
                self.to_archive.append(behavior)

        return avg_dist

    def post_evaluation(self):
        #print("archive=", len(self.archive), "threshold=", self.archive_threshold)
        if len(self.to_archive) == 0:
            self.archive_under += 1

            if self.archive_under == self.archive_under_threshold:
                self.archive_threshold /= self.archive_threshold_factor
                #print("lowering threshold to", self.archive_threshold)

                if self.archive_threshold < self.archive_threshold_min:
                    self.archive_threshold = self.archive_threshold_min

                self.archive_under = 0
        else:
            self.archive_under = 0

            if len(self.to_archive) > self.archive_over_threshold:
                self.archive_threshold *= self.archive_threshold_factor
                #print("increasing threshold to", self.archive_threshold)

        self.archive.extend(self.to_archive)
        self.to_archive.clear()

    def _distance_between(self, b1, b2):
        distance = np.linalg.norm(b1 - b2)

        return distance

    def _contains_similar(self, population, behavior):
        for b in population:
            dist = self._distance_between(behavior, b)
            #print("dist=", dist, "threshold=", self.archive_threshold, "<", (dist < self.archive_threshold))
            if dist < self.archive_threshold:
                return True

        return False
