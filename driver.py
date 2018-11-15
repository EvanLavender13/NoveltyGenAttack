from gen_attack import GenAttack

gen = GenAttack(dist_delta=1, step_size=1, pop_size=1, cx_prob=1, mut_prob=1)

gen.attack(orig_img=None, target=None, num_gen=1)
