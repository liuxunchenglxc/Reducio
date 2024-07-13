def score_sr(psnr, ssim, runtime):
    score = (2 ** (2 * (psnr - 26))) / runtime
    return score
