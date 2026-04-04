import numpy as np

def cosine_similarity_weighted(u, v, alpha=10):
    mask = (u != 0) & (v != 0)
    overlap = np.sum(mask)

    if overlap == 0:
        return 0

    u_common = u[mask]
    v_common = v[mask]

    sim = np.dot(u_common, v_common) / (
            np.linalg.norm(u_common) * np.linalg.norm(v_common)
    )

    # significance weighting
    weight = overlap / (overlap + alpha)

    return sim * weight

def cosine_with_all(target, matrix):
    sims = []
    for user in matrix:
        sims.append(cosine_similarity_weighted(target, user))
    return np.array(sims)