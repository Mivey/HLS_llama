# 2025-11-27T21:06:33.187550443
import vitis

client = vitis.create_client()
client.set_workspace(path="forward")

advanced_options = client.create_advanced_options_dict(user_dtsi="/home/lolwut/proj/common/TE0950.dtsi",dt_zocl="1",dt_overlay="0")

platform = client.create_platform_component(name = "TE0950_plat",hw_design = "$COMPONENT_LOCATION/../hw/TE_0950_ex_emb_pl_hw.xsa",os = "linux",cpu = "psv_cortexa72",domain_name = "linux_psv_cortexa72",advanced_options = advanced_options)

