
//      ---- LIGHTBOX_FUNCTIONALITY ----
/**
 * This function is triggered when a flower is 
 * clicked.
 * src (The image source) is passed to the function
 */
function openLightbox( src )
{
    /** ---- document.getElementById('lightbox-img').src = src; ----
     * Upates the "src" attribute of the image inside 
     * the lightbox, to match the clicked image.
     */
    document.getElementById( 'lightbox-img' ).src = src;
    /** Bottom Code: Displays. Makes the lightbox visible */
    document.getElementById( 'lightbox' ).style.display = 'block';
}

function closeLightbox()
{
    // Hides the "lightbox"
    document.getElementById( 'lightbox' ).style.display = 'none';
}

/** 
 * This listens for the scroll event.
 */
window.addEventListener( 'scroll', function() 
{
    /** "window.scrollY > 300": checks if the user has 
     * scrolled more than 300 pixels down. If true, 
     * document.getElementById('scrollTop').style.display = 
     * 'block'; makes the button visible. If false, 
     * document.getElementById('scrollTop').style.display = 
     * 'none'; hides the button. */
    this.document.getElementById( 'scrollTop' ).style.display = this.window.scrollY > 300 ? 'block' : 'none';
});

/** This function smoothly scrolls the page back 
 * to the top */
function scrollToTop()
{
    window.scrollTo( { top: 0, behavior: 'smooth' } );
}