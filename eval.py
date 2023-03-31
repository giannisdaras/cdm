"""Single evaluation file (for generating and measuring Inception/FID."""
import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist
from training import dataset
import scipy.linalg
import wandb
import time


def sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    sampler_seed=42):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next


def inpainting_sampler(
    net, y, mask, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    sampler_seed=42):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        
        # add noise to y to bring it to the appropriate noise level
        # noisy_y = y + t_next
        noisy_y = y + t_next * d_prime

        # project back
        x_next = (1 - mask) * x_next + mask * noisy_y
        
    return x_next


#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------



@click.command()
@click.option('--network', 'network_loc',  help='Location of the folder where the network is stored', metavar='PATH|URL',                      type=str, required=True)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         is_flag=True)
@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)

@click.option('--experiment_name', help="Name of the experiment to log to wandb", type=str, required=True)
@click.option('--wandb_id', help='Id of wandb run to resume', type=str, default='')
@click.option('--ref', 'ref_path',      help='Dataset reference statistics ', metavar='NPZ|URL',    type=str, required=True)
@click.option('--num', 'num_expected',  help='Number of images to use', metavar='INT',              type=click.IntRange(min=2), default=50000, show_default=True)
@click.option('--seed',                 help='Random seed for selecting the images', metavar='INT', type=int, default=0, show_default=True)

@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0))
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0))
@click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0), default=7, show_default=True)
@click.option('--S_churn', 'S_churn',      help='Stochasticity strength', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',          help='Stoch. min noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',          help='Stoch. max noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',      help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=1, show_default=True)

@click.option('--solver',                  help='Ablate ODE solver', metavar='euler|heun',                          type=click.Choice(['euler', 'heun']))
@click.option('--disc', 'discretization',  help='Ablate time step discretization {t_i}', metavar='vp|ve|iddpm|edm', type=click.Choice(['vp', 've', 'iddpm', 'edm']))
@click.option('--schedule',                help='Ablate noise schedule sigma(t)', metavar='vp|ve|linear',           type=click.Choice(['vp', 've', 'linear']))
@click.option('--scaling',                 help='Ablate signal scaling s(t)', metavar='vp|none',                    type=click.Choice(['vp', 'none']))


def main(network_loc, outdir, subdirs, seeds, class_idx, max_batch_size,
    experiment_name, wandb_id, ref_path, num_expected, seed, device=torch.device('cuda'), **sampler_kwargs):
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    # we want to make sure that each gpu does not get more than batch size.
    # Hence, the following measures how many batches are going to be per GPU.
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    

    dist.print0(f"The algorithm will run for {num_batches} batches --  {len(seeds)} images of batch size {max_batch_size}")
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    # the following has for each batch size allocated to this GPU, the indexes of the corresponding images.
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]
    batches_per_process = len(rank_batches)
    dist.print0(f"This process will get {len(rank_batches)} batches.")

    if dist.get_rank() == 0:
        wandb.init(
            project="cdm",
            name=experiment_name,
            id=wandb_id if wandb_id else None,
            resume="must" if wandb_id else False
        )
        dist.print0("Initialized wandb")

    eval_index = 0  # keeps track of how many checkpoints we have evaluated so far
    while True:
        # find all *.pkl files in the folder network_loc and sort them
        files = os.listdir(network_loc)
        # Filter the list to include only "*.pkl" files
        pkl_files = [f for f in files if f.endswith('.pkl')]
        # Sort the list of "*.pkl" files
        sorted_pkl_files = sorted(pkl_files)[eval_index:]


        checkpoint_numbers = []
        for curr_file in sorted_pkl_files:
            checkpoint_numbers.append(int(curr_file.split('-')[-1].split('.')[0]))
        checkpoint_numbers = np.array(checkpoint_numbers)

        if len(sorted_pkl_files) == 0:
            dist.print0("No new checkpoint found! Going to sleep for 1min!")
            time.sleep(60)
            dist.print0("Woke up!")
        
        for checkpoint_number, checkpoint in zip(checkpoint_numbers, sorted_pkl_files):
            # Rank 0 goes first.
            if dist.get_rank() != 0:
                torch.distributed.barrier()

            network_pkl = os.path.join(network_loc, checkpoint)
            # Load network.
            dist.print0(f'Loading network from "{network_pkl}"...')
            with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
                net = pickle.load(f)['ema'].to(device)
            dist.print0(f'Network loaded!')

            # Other ranks follow.
            if dist.get_rank() == 0:
                torch.distributed.barrier()

            # Loop over batches.
            dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
            batch_count = 1
            for batch_seeds in tqdm.tqdm(rank_batches, disable=dist.get_rank() != 0):
                dist.print0(f"Waiting for the green light to start generation for {batch_count}/{batches_per_process}")
                # don't move to the next batch until all nodes have finished their current batch
                torch.distributed.barrier()
                dist.print0("Others finished. Good to go!")
                batch_size = len(batch_seeds)
                if batch_size == 0:
                    continue

                # Pick latents and labels.
                rnd = StackedRandomGenerator(device, batch_seeds)
                latents = rnd.randn([batch_size, 3, net.img_resolution, net.img_resolution], device=device)
                class_labels = None
                if net.label_dim:
                    class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[batch_size], device=device)]
                if class_idx is not None:
                    class_labels[:, :] = 0
                    class_labels[:, class_idx] = 1

                # Generate images.
                sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
                dist.print0(f"Running the sampler")
                images = sampler(net, latents, class_labels, randn_like=rnd.randn_like, sampler_seed=batch_seeds, **sampler_kwargs)
                dist.print0(f"Got the images, saving them!")
                # Save images.
                images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
                for seed, image_np in zip(batch_seeds, images_np):
                    image_dir = os.path.join(outdir, str(checkpoint_number), f'{seed-seed%1000:06d}') if subdirs else os.path.join(outdir, str(checkpoint_number))
                    os.makedirs(image_dir, exist_ok=True)
                    image_path = os.path.join(image_dir, f'{seed:06d}.png')
                    if image_np.shape[2] == 1:
                        PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
                    else:
                        PIL.Image.fromarray(image_np, 'RGB').save(image_path)
                dist.print0("Saved images!")
                batch_count += 1
                
            dist.print0(f"Node finished generation for {checkpoint_number}")
            dist.print0("waiting for others to finish..")
            torch.distributed.barrier()
            dist.print0("Others finished generation..")

            # Rank 0 goes first.
            if dist.get_rank() != 0:
                torch.distributed.barrier()

            calc(os.path.join(outdir, str(checkpoint_number)), ref_path, num_expected, seed, max_batch_size, image_size=image_np.shape[-2])
            torch.distributed.barrier() 
            eval_index += 1
            dist.print0('Done.')

#----------------------------------------------------------------------------


def calculate_inception_stats(
    image_path, num_expected=None, seed=0, max_batch_size=64,
    num_workers=3, prefetch_factor=2, device=torch.device('cuda'),
):

    # Load Inception-v3 model.
    # This is a direct PyTorch translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    dist.print0('Loading Inception-v3 model...')
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True)
    inception_kwargs = dict(no_output_bias=True) # Match the original implementation by not applying bias in the softmax layer.
    feature_dim = 2048
    with dnnlib.util.open_url(detector_url, verbose=(dist.get_rank() == 0)) as f:
        detector_net = pickle.load(f).to(device)

    # List images.
    dist.print0(f'Loading images from "{image_path}"...')
    dataset_obj = dataset.ImageFolderDataset(path=image_path, max_size=num_expected, random_seed=seed)
    if num_expected is not None and len(dataset_obj) < num_expected:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but expected at least {num_expected}')
    if len(dataset_obj) < 2:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but need at least 2 to compute statistics')

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide images into batches.
    num_batches = ((len(dataset_obj) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.arange(len(dataset_obj)).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]
    data_loader = torch.utils.data.DataLoader(dataset_obj, batch_sampler=rank_batches, num_workers=num_workers, prefetch_factor=prefetch_factor)

    # Accumulate statistics.
    dist.print0(f'Calculating statistics for {len(dataset_obj)} images...')
    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)

    all_features = []

    for images, _labels in tqdm.tqdm(data_loader, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        if images.shape[0] == 0:
            continue
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])

        # fid 
        features = detector_net(images.to(device), **detector_kwargs).to(torch.float64)
        mu += features.sum(0)
        sigma += features.T @ features

        # inception
        inception_features = torch.clamp(detector_net(images.to(device), **inception_kwargs), min=1e-6, max=1.0)
        all_features.append(inception_features.to(torch.float64))


    all_features = torch.cat(all_features, dim=0).reshape(-1, inception_features.shape[-1]).to(torch.float64)
    dist.print0("Features computed locally.")
    dist.print0("Wait for all others to finish before gathering...")
    torch.distributed.barrier()
    dist.print0("Gathering process started...")

    all_features_list = [torch.ones_like(all_features) for _ in range(dist.get_world_size())]
    torch.distributed.all_gather(all_features_list, all_features)
    all_features_gathered = torch.cat(all_features_list, dim=0)
    
    gen_probs = all_features_gathered.reshape(-1, all_features.shape[-1]).cpu().numpy()
    dist.print0(f"{gen_probs.shape}, {gen_probs.min()}, {gen_probs.max()}")
    dist.print0("Computing KL...")
    kl = gen_probs * (np.log(gen_probs) - np.log(np.mean(gen_probs, axis=0, keepdims=True)))
    kl = np.mean(np.sum(kl, axis=1))
    dist.print0("KL computed...")
    inception_score = np.mean(np.exp(kl))
    dist.print0(f"Inception score: {inception_score}")


    # Calculate grand totals.
    torch.distributed.all_reduce(mu)
    torch.distributed.all_reduce(sigma)
    mu /= len(dataset_obj)
    sigma -= mu.ger(mu) * len(dataset_obj)
    sigma /= len(dataset_obj) - 1



    return mu.cpu().numpy(), sigma.cpu().numpy(), inception_score

#----------------------------------------------------------------------------

def calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))

    

def calc(image_path, ref_path, num_expected, seed, batch, num_rows=8, num_cols=8, image_size=32):
    """Calculate Inception/FID for a given set of images."""
    assert num_rows * num_cols <= num_expected, "You need to save more images."
    dist.print0("Starting FID calculation...")
    dist.print0(f'Loading dataset reference statistics from "{ref_path}"...')
    ref = None
    if dist.get_rank() == 0:
        with dnnlib.util.open_url(ref_path) as f:
            ref = dict(np.load(f))
        
    checkpoint_index = int(image_path.split('/')[-1])

    if dist.get_rank() == 0:
        dist.print0("Creating image collage...")
        # create a blank image to hold the grid
        grid_image = PIL.Image.new('RGB', (num_cols * image_size, num_rows * image_size))
        for i in range(num_rows):
            for j in range(num_cols):
                index = i * num_cols + j
                sample_image_path = os.path.join(image_path, f"{index:06d}.png")
                img_array = np.array(PIL.Image.open(sample_image_path))
                img = PIL.Image.fromarray(img_array)
                grid_image.paste(img, (i * image_size, j * image_size))
        dist.print0("Finished collage creation")
    

    mu, sigma, inception = calculate_inception_stats(image_path=image_path, num_expected=num_expected, seed=seed, max_batch_size=batch)
    dist.print0(f'Calculating FID for {image_path}...')
    if dist.get_rank() == 0:
        fid = calculate_fid_from_inception_stats(mu, sigma, ref['mu'], ref['sigma'])
        dist.print0(f"FID: {fid:g}")

    torch.distributed.barrier()
    if dist.get_rank() == 0:
        wandb.log({"FID": fid, "Inception": inception, "image_grid": wandb.Image(grid_image)}, step=checkpoint_index, commit=True)
    dist.print0("Computed FID and logged it.")

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------