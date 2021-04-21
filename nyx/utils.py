import jax
import jax.numpy as jnp


@jax.jit
def haversine(lon1, lat1, lon2, lat2, R=6371):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(jnp.radians, [lon1, lat1, lon2, lat2])
    dlon = jnp.subtract(lon2, lon1)
    dlat = jnp.subtract(lat2, lat1)
    a = jnp.add(
        np.power(jnp.sin(jnp.divide(dlat, 2)), 2),
        jnp.multiply(
            jnp.cos(lat1),
            jnp.multiply(jnp.cos(lat2), jnp.power(jnp.sin(jnp.divide(dlon, 2)), 2)),
        ),
    )
    c = jnp.multiply(2, jnp.arcsin(jnp.sqrt(a)))
    km = jnp.multiply(R, c)
    return km


@jax.jit
def lon_lat_to_cartesian(lon, lat, R=6371):
    """
    Calculates lon, lat coordinates of a point on a sphere with radius R = 6371 km for earth.
    """
    lon_r = jnp.radians(lon)
    lat_r = jnp.radians(lat)
    x = R * jnp.cos(lat_r) * jnp.cos(lon_r)
    y = R * jnp.cos(lat_r) * jnp.sin(lon_r)
    z = R * jnp.sin(lat_r)
    return x, y, z


@jax.jit
def to_cartesian(X):
    """
    X is ordered as lon, lat and optional additional variables
    """
    (
        c1_,
        c2_,
        c3_,
    ) = lon_lat_to_cartesian(lon=X[:, 0], lat=X[:, 1])
    Xt = jnp.column_stack([c1_, c2_, c3_, X[:, 2:]])
    return Xt


@jax.jit
def from_cartesian(X):
    R = 6371
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    lat = jnp.degrees(jnp.arcsin(z / R))
    lon = jnp.degrees(jnp.arctan2(y, x))
    return jnp.column_stack([lon, lat, X[:, 3:]])
