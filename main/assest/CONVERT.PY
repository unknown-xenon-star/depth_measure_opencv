from moviepy import VideoFileClip


def convert_video_to_width_1280(input_path, output_path):
    try:
        # Load video
        clip = VideoFileClip(input_path)

        # Resize while maintaining aspect ratio
        resized_clip = clip.resized(width=1280)

        # Save video
        resized_clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac"
        )

        # Close clips
        clip.close()
        resized_clip.close()

        print("Conversion completed!")

    except Exception as e:
        print("Error:", e)


# Example
convert_video_to_width_1280(
    "car002.m2ts",
    "video_width_1280.mp4"
)