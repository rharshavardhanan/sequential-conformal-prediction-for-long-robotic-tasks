"""Watch the closed loop. --policy expert | bcrnn | act

The expert is the reference: it closes the loop and builds the tower. The
learned arms close the same loop through the same controller and do not
complete the task -- see docs/tower-findings.md.
"""

import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="expert",
                    choices=["expert", "bcrnn", "act"])
    ap.add_argument("--ckpt-dir", default="data/tower/rel")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=800)
    ap.add_argument("--res", type=int, default=460)
    ap.add_argument("--save", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import matplotlib.pyplot as plt

    from scripts.tower_collect import make_env, observation
    from scripts.tower_control import TowerExpert, joint_action

    env = make_env(seed=args.seed, render=True, res=args.res)
    if args.policy == "expert":
        pol, name = TowerExpert(env), "Scripted expert"
    elif args.policy == "bcrnn":
        from scripts.tower_policies import BCRNNPolicy
        pol = BCRNNPolicy(f"{args.ckpt_dir}/tower_bcrnn.pt"); name = "BC-RNN (cheap)"
    else:
        from scripts.tower_policies import ACTPolicy
        pol = ACTPolicy(f"{args.ckpt_dir}/tower_act.pt"); name = "ACT (expensive)"

    fig, (axr, axb) = plt.subplots(1, 2, figsize=(12.5, 6.2),
                                   gridspec_kw={"width_ratios": [1.5, 1]})
    fig.canvas.manager.set_window_title("Tower Creation - closed loop")
    axr.axis("off"); axb.axis("off")
    im = None
    frames = []

    for ep in range(args.episodes):
        o = env.reset()
        pol.reset()
        for t in range(args.max_steps):
            if args.policy == "expert":
                q, g = pol.act(); g = 1.0 if g > 0 else 0.0
            else:
                q, g = pol.act(observation(env))
            o, _, _, _ = env.step(joint_action(env, q, -1.0 if g < .5 else 1.0))

            if t % 2 == 0:
                img = o["frontview_image"][::-1]
                im = axr.imshow(img) if im is None else (im.set_data(img) or im)
                h = env.tower_height()
                axr.set_title(f"{name}   ep {ep+1}/{args.episodes}   step {t}",
                              fontsize=14, fontweight="bold")
                axb.clear(); axb.axis("off")
                axb.text(.02, .92, "TOWER", fontsize=13, fontweight="bold",
                         transform=axb.transAxes)
                for lvl in range(3, 0, -1):
                    on = h >= lvl
                    axb.barh(.80 - (3 - lvl) * .09, .5 if on else .06, height=.055,
                             left=.05, transform=axb.transAxes,
                             color="#2e7d32" if on else "#ddd")
                    axb.text(.60, .79 - (3 - lvl) * .09, f"layer {lvl}",
                             fontsize=11, transform=axb.transAxes,
                             color="#2e7d32" if on else "#aaa")
                axb.text(.02, .46, f"height   {h} / 3", fontsize=13,
                         family="monospace", transform=axb.transAxes,
                         fontweight="bold")
                axb.text(.02, .38, f"gripper  {'CLOSED' if g > .5 else 'OPEN'}",
                         fontsize=12, family="monospace", transform=axb.transAxes)
                axb.text(.02, .30,
                         f"success  {'YES' if env._check_success() else 'not yet'}",
                         fontsize=12, family="monospace", transform=axb.transAxes,
                         color="#2e7d32" if env._check_success() else "#555")
                axb.text(.02, .16, "policy -> 7 joint targets + grip",
                         fontsize=10, transform=axb.transAxes, color="#777")
                axb.text(.02, .10, "-> rate-limited delta -> MuJoCo",
                         fontsize=10, transform=axb.transAxes, color="#777")
                if args.save:
                    fig.canvas.draw()
                    frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
                else:
                    plt.pause(.001)
                    if not plt.fignum_exists(fig.number):
                        break
            if env._check_success():
                break
        print(f"  ep {ep+1}: height {env.tower_height()}/3   "
              f"success {env._check_success()}   steps {t+1}")

    if args.save and frames:
        import imageio
        imageio.mimsave(args.save, frames, fps=25)
        print(f"wrote {args.save}")
    env.close()


if __name__ == "__main__":
    main()
