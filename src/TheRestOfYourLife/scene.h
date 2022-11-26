#ifndef SCENE_H
#define SCENE_H
//==============================================================================================
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================

#include "rtweekend.h"

#include "camera.h"
#include "hittable_list.h"

inline double luminance(const color& c)
{
    return c.x() * 0.2126 + c.y() * 0.7152 + c.z() * 0.0722;
}

class scene {
  public:
    void render() {
        const int image_height = static_cast<int>(image_width / aspect_ratio),
            n_sample_paths = samples_per_pixel * image_width * image_height;

        cam.initialize(aspect_ratio);

        const unsigned n_initial_paths = image_height * image_width;
        std::vector<double> initial_path_distribution(n_initial_paths);
        double partial_sum = 0.;
        for (unsigned i = 0; i < n_initial_paths; ++i) {
            metropolis_sampler sampler{ i };
            sampler();

            auto s = sampler.get(), t = sampler.get();
            ray r = cam.get_ray(s, t, sampler);
            partial_sum += luminance(ray_color(r, max_depth, sampler));
            initial_path_distribution[i] = partial_sum;
        }

        // Choose the initial path.
        std::default_random_engine e;
        std::uniform_real_distribution<double> u{ 0. };
        auto iter = std::upper_bound(
            initial_path_distribution.begin(),
            initial_path_distribution.end(),
            u(e) * initial_path_distribution.back()
        );
        auto seed = iter - initial_path_distribution.begin() - (iter == initial_path_distribution.end());
        metropolis_sampler sampler{ static_cast<unsigned>(seed) };

        std::vector<color> frame_buffer{ static_cast<unsigned>(image_height * image_width) };

        sampler();
        auto s = sampler.get(), t = sampler.get();
        ray r = cam.get_ray(s, t, sampler);
        auto current_spectrum = ray_color(r, max_depth, sampler);
        auto current_luminance = luminance(current_spectrum);
        auto current_pixel = &frame_buffer[static_cast<int>(image_height * t) * image_width + static_cast<int>(image_width * s)];

        for (int i = 0; i < n_sample_paths; ++i) {
            sampler();

            auto s = sampler.get(), t = sampler.get();
            ray r = cam.get_ray(s, t, sampler);
            auto proposed_spectrum = ray_color(r, max_depth, sampler);

            auto proposed_luminance = luminance(proposed_spectrum);
            // Handle the case where current_luminance == 0.
            auto transition_probability = current_luminance <= proposed_luminance ? 1. : proposed_luminance / current_luminance;

            auto& proposed_pixel = frame_buffer[static_cast<int>(image_height * t) * image_width + static_cast<int>(image_width * s)];
            if (proposed_luminance)
                proposed_pixel += transition_probability * proposed_spectrum / proposed_luminance;
            *current_pixel += (1 - transition_probability) * current_spectrum / current_luminance;

            if (u(e) < transition_probability) {
                current_spectrum = proposed_spectrum;
                current_luminance = proposed_luminance;
                current_pixel = &proposed_pixel;
                sampler.accept();
            }
            else
                sampler.reject();
        }

        std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

        const auto scale_factor = initial_path_distribution.back() / initial_path_distribution.size() / samples_per_pixel;
        for (const auto& p : frame_buffer)
            write_color(std::cout, p, scale_factor);

        std::clog << "\rDone.                 \n";
    }

  public:
    hittable_list world;
    hittable_list lights;
    camera        cam;

    double aspect_ratio      = 1.0;
    int    image_width       = 100;
    int    samples_per_pixel = 10;
    int    max_depth         = 20;
    color  background        = color(0,0,0);

  private:
    color ray_color(const ray& r, int depth, metropolis_sampler& sampler) {
        hit_record rec;

        // If we've exceeded the ray bounce limit, no more light is gathered.
        if (depth <= 0)
            return color(0,0,0);

        // If the ray hits nothing, return the background color.
        if (!world.hit(r, interval(0.001, infinity), rec))
            return background;

        scatter_record srec{ sampler };
        color color_from_emission = rec.mat->emitted(r, rec, rec.u, rec.v, rec.p);

        if (!rec.mat->scatter(r, rec, srec))
            return color_from_emission;

        if (srec.skip_pdf) {
            return srec.attenuation * ray_color(srec.skip_pdf_ray, depth-1, sampler);
        }

        auto light_ptr = make_shared<hittable_pdf>(lights, rec.p);
        mixture_pdf p(light_ptr, srec.pdf_ptr);

        ray scattered = ray(rec.p, p.generate(sampler), r.time());
        auto pdf_val = p.value(scattered.direction());

        double scattering_pdf = rec.mat->scattering_pdf(r, rec, scattered);

        color color_from_scatter =
            (srec.attenuation * scattering_pdf * ray_color(scattered, depth-1, sampler)) / pdf_val;

        return color_from_emission + color_from_scatter;
    }
};


#endif
