"""Controller adapter factory."""


def build_controller(name: str, log_every=1, always_forward=True):
    if name == "vigem":
        from controllers.vigem_controller import ViGEmController

        return ViGEmController(always_forward=always_forward)
    if name == "pydirect":
        from controllers.pydirect_controller import PyDirectController

        return PyDirectController()
    if name == "dry-run":
        from controllers.dry_run_controller import DryRunController

        return DryRunController(log_every=log_every)
    raise ValueError(f"Unknown controller: {name}")
