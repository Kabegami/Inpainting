from machine_learning import *
import click

def noise_inpainting(fname, bruit, step, p_h):
    M, h, l = read_im_tensor(fname+".png")
    name = fname.split("/")[-1]
    alpha = get_alpha('a_'+name)
    noise(M, bruit, p_h)
    display_img_tensor(M)
    N = correct_img(M, p_h, step, alpha)
    display_img_tensor(N)

def hole_inpainting_heuristique(fname,i,j, h1, h2, step, p_h):
    M, h, l = read_im_tensor(fname+".png")
    name = fname.split("/")[-1]
    alpha = get_alpha('a_'+name)
    delete_rect(M, i,j,h1,h2)
    display_img_tensor(M)
    N = correctImgHeuristique(M, p_h, step, alpha)
    display_img_tensor(N)

def hole_nearest_neighbour(fname, i, j, h1, h2, step, p_h):
    M, h, l = read_im_tensor(fname+".png")
    name = fname.split("/")[-1]
    alpha = get_alpha('a_'+name)
    delete_rect(M, i,j,h1,h2)
    display_img_tensor(M)
    N = correct_hole(M, p_h, step, alpha)
    display_img_tensor(N)

def calculate_alpha(fname, bruit, step, p_h, n=5):
    M, h, l = read_im_tensor(fname+".png")
    name = fname.split("/")[-1]
    noise(M, bruit, p_h)
    alpha = findAlphaNoise(M, p_h, step, 'a_'+name, n)

@click.command()
@click.option('--method',default="noise", type=click.Choice(["noise", "heuristique", "alpha", "neighbour"]))
@click.option('--image_name',default="lena")
@click.option('--noise_pct',default=0.2)
def main(method, image_name, noise_pct):
    """ noise : corrige une image bruité de manière uniforme.
        
        heuristique : corriger une image avec un carré manquant à l'aide de l'heuristique de confidence 

        neighbour : corrige une image avec un carré manquant en prennant les pixels avec le plus de voisins vivant et en faisant tous le contour avant de mettre a jour


        calculate_alpha : calcule le alpha pour une image et un bruit donné
    """
    if method == "noise":
        noise_inpainting('../data/images/'+image_name, noise_pct, 6, 3)
        return
    if method == "heuristique":
        hole_inpainting_heuristique('../data/images/' + image_name, 20, 20, 50, 50, 6, 15)
        return
    if method == "alpha":
        calculate_alpha("../data/images/"+image_name, noise_pct, 6, 3, 5)
        return
    if method == "neighbour":
        hole_nearest_neighbour('../data/images/' + image_name, 20, 20, 50, 50, 6, 15)
        return


if __name__ == "__main__":
    main()
