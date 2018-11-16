from gen_attack import GenAttack

gen = GenAttack(dist_delta=1, step_size=1, pop_size=1, cx_prob=1, mut_prob=1, model=None)

gen.attack(orig_img=None, target=None, pop_size=25, num_gen=1)
