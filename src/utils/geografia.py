from math import radians, sin, cos, sqrt, atan2


def calcular_distancia(lat1, lon1, lat2, lon2):
    """
    Calcula a distância entre dois pontos em quilômetros
    usando a fórmula de Haversine.
    """

    raio_terra = 6371.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)

    lat2 = radians(lat2)
    lon2 = radians(lon2)

    diferenca_lat = lat2 - lat1
    diferenca_lon = lon2 - lon1

    a = (
        sin(diferenca_lat / 2) ** 2
        + cos(lat1)
        * cos(lat2)
        * sin(diferenca_lon / 2) ** 2
    )

    c = 2 * atan2(
        sqrt(a),
        sqrt(1 - a)
    )

    return raio_terra * c