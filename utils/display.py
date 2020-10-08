from IPython.display import HTML

def video_display(video):
    return HTML("""<video controls>
        <source src={} type="video/webm">
        </video>""".format(video))