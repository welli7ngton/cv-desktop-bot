import cv2

from os import makedirs, remove
from time import sleep
from pyautogui import click, screenshot


class CVisionBot:
    def __init__(self, resources_base_path: str, *, images_folder_name: str = 'images', default_sleep: int = 5):
        self.resources_base_path = fr'{resources_base_path}\{images_folder_name}'
        self.target = fr'{self.resources_base_path}\target'
        self.output = fr'{self.resources_base_path}\output'
        self.screenshot = fr'{self.resources_base_path}\screenshot'

        self.default_sleep = default_sleep

        for folder in [self.output, self.target, self.screenshot]:
            makedirs(folder, exist_ok=True)

    @classmethod
    def _get_item_location(cls, template_path: str, screenshot_path: str, output_path: str):
        img = cv2.imread(screenshot_path, cv2.IMREAD_GRAYSCALE)
        assert img is not None, "file could not be read, check with os.path.exists()"
        img_color = cv2.imread(screenshot_path)
        assert img_color is not None, "file could not be read, check with os.path.exists()"

        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        assert template is not None, "file could not be read, check with os.path.exists()"
        w, h = template.shape[::-1]

        methods = ['TM_CCOEFF']

        for methd in methods:
            method = getattr(cv2, methd)

            # Aplica a correspondência de template
            res = cv2.matchTemplate(img, template, method)
            _, _, min_loc, max_loc = cv2.minMaxLoc(res)

            # Escolhe a localização baseada no método
            top_left = max_loc if methd in ['TM_CCOEFF', 'TM_CCOEFF_NORMED'] else min_loc

            # Calcula o centro do retângulo detectado
            center_x = top_left[0] + w // 2
            center_y = top_left[1] + h // 2

            # Desenha o retângulo na imagem original colorida
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(img_color, top_left, bottom_right, (0, 255, 0), 2)

            # Salva a imagem com o retângulo desenhado
            cv2.imwrite(output_path, img_color)
            print(f"Imagem de saída salva em: {output_path}")

            return center_x, center_y

    def find_and_click(self, step: int, name: str) -> tuple[int, int]:
        """
        Tira um screenshot, encontra a posição do item na tela e clica nele.
        Args:
            step (int): Número referente a contagem de passos da automação, é útil para se situar
            onde a automação está no processo, esse parâmetro organiza todos os prints em sequência.
            name (str): Nome da imagem alvo que foi salva como referência de onde o robô deve buscar
            as coordenadas.
        """
        screenshot_path = fr"{self.screenshot}\{step}_.png"
        screenshot(screenshot_path)

        x, y = __class__._get_item_location(
            fr"{self.target}\{step}_{name}.png",
            screenshot_path,
            fr"{self.output}\{step}_.png",
        )
        click(x, y)
        sleep(self.default_sleep)
        remove(screenshot_path)
        return x, y
