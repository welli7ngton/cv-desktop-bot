import cv2
import logging

from os import makedirs, remove
from time import sleep
from pyautogui import click, screenshot


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

_logger = logging.getLogger(__name__)


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
    def _get_item_location(cls, template_path: str, screenshot_path: str, output_path: str, precisao: float = 0.7):
        img = cv2.imread(screenshot_path, cv2.IMREAD_GRAYSCALE)
        assert img is not None, "file could not be read, check with os.path.exists()"
        img_color = cv2.imread(screenshot_path)
        assert img_color is not None, "file could not be read, check with os.path.exists()"

        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        assert template is not None, "file could not be read, check with os.path.exists()"
        w, h = template.shape[::-1]

        method = cv2.TM_CCOEFF_NORMED  # método que dá resultado de 0 a 1 (ideal pra comparar precisão)
        res = cv2.matchTemplate(img, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        _logger.info(f"Precisão encontrada: {max_val}")

        if max_val < precisao:
            _logger.warning(f"Imagem não encontrada com precisão mínima de {precisao}")
            return None, None

        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        center_x = top_left[0] + w // 2
        center_y = top_left[1] + h // 2

        cv2.rectangle(img_color, top_left, bottom_right, (0, 255, 0), 2)
        cv2.imwrite(output_path, img_color)
        _logger.info(f"Imagem de saída salva em: {output_path}")

        return center_x, center_y

    def find_and_click(self, step: int, name: str, precisao: float = 0.7) -> tuple[int, int]:
        """
        Tira um screenshot, encontra a posição do item na tela e clica nele.
        Args:
            step (int): Número referente a contagem de passos da automação, é útil para se situar
            onde a automação está no processo, esse parâmetro organiza todos os _logger.infos em sequência.
            name (str): Nome da imagem alvo que foi salva como referência de onde o robô deve buscar
            as coordenadas.
        """
        screenshot_path = fr"{self.screenshot}\{step}_.png"
        screenshot(screenshot_path)

        x, y = __class__._get_item_location(
            fr"{self.target}\{step}_{name}.png",
            screenshot_path,
            fr"{self.output}\{step}_.png",
            precisao
        )

        if x is not None and y is not None:
            click(x, y)
            sleep(self.default_sleep)
        else:
            raise Exception('Clique ignorado por falta de precisao.')

        remove(screenshot_path)
        return x, y
