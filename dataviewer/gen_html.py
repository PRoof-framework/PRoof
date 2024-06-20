from jinja2 import Environment, PackageLoader, select_autoescape
from shutil import copytree
import os
from pathlib import Path


def _print_error(page: str):
    import traceback
    traceback.print_exc()
    print("ERROR: page %s generate failed" % page)


def gen_report(collected_data: dict, output_path: str) -> None:
    env = Environment(
        loader=PackageLoader("gen_html", "html_templates"),
        autoescape=select_autoescape()
    )

    templates_path = Path(__file__).parent / "html_templates"

    # copy assets
    assets_src = str(templates_path / "assets")
    assets_dst = str(Path(output_path) / "assets")
    copytree(assets_src, assets_dst, dirs_exist_ok=True)

    # generate html from templates
    for template_file in templates_path.glob("*.html"):
        template = env.get_template(template_file.name)
        if template_file.stem.endswith("META_BS"):
            # meta page, generate one page for each batch_size
            for batch_size in collected_data['model']['bench']['batch_size_list']:
                out_file = Path(output_path) / template_file.name.replace("META_BS", str(batch_size))
                try:
                    out_file.write_text(template.render(data=collected_data, batch_size=batch_size))
                except Exception as e:
                    _print_error(out_file)

        else:
            # normal single page
            out_file = Path(output_path) / template_file.name
            try:
                out_file.write_text(template.render(data=collected_data))
            except Exception as e:
                _print_error(out_file)
