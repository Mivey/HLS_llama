# 2025-11-27T21:13:38.196919676
import vitis

client = vitis.create_client()
client.set_workspace(path="forward")

platform = client.get_component(name="TE0950_plat")
domain = platform.add_domain(cpu = "ai_engine",os = "aie_runtime",name = "aie",display_name = "aie",generate_dtb = False,hw_boot_bin = "")

domain = platform.get_domain(name="linux_psv_cortexa72")

status = domain.generate_bif()

status = domain.set_boot_dir(path="$COMPONENT_LOCATION/../xilinx-versal-common-v2025.2")

status = domain.set_sd_dir(path="$COMPONENT_LOCATION/../data")

status = domain.set_sd_dir(path="../data")

status = domain.set_boot_dir(path="$COMPONENT_LOCATION/../xilinx-versal-common-v2025.2")

status = domain.set_sd_dir(path="$COMPONENT_LOCATION/../data")

status = platform.build()

client.sync_git_example_repo(name="vitis_examples")

client.sync_git_example_repo(name="vitis_libraries")

proj = client.create_sys_project(name="system_project", platform="$COMPONENT_LOCATION/../TE0950_plat/export/TE0950_plat/TE0950_plat.xpfm", template="empty_accelerated_application" , build_output_type="xsa")

proj = client.get_sys_project(name="system_project")

status = proj.add_container(name="data_f")

proj = proj.add_component(name="dataflow_forward", container_name="data_f")

comp = client.create_app_component(name="test_app",platform = "$COMPONENT_LOCATION/../TE0950_plat/export/TE0950_plat/TE0950_plat.xpfm",domain = "linux_psv_cortexa72")

comp = client.get_component("test_app")

status = comp.set_sysroot(sysroot="/home/lolwut/proj/2025/plat/forward/xilinx-versal-common-v2025.2/sysroots/cortexa72-cortexa53-amd-linux")

proj = proj.add_component(name="test_app")

proj.build(target = "hw",build_comps = False)

