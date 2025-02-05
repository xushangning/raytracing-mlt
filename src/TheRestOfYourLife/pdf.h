#ifndef PDF_H
#define PDF_H
//==============================================================================================
// Originally written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================

#include "rtweekend.h"

#include "hittable_list.h"
#include "onb.h"


class pdf {
  public:
    virtual ~pdf() {}

    virtual double value(const vec3& direction) const = 0;
    virtual vec3 generate(metropolis_sampler& sampler) const = 0;
};


class cosine_pdf : public pdf {
  public:
    cosine_pdf(const vec3& w) { uvw.build_from_w(w); }

    double value(const vec3& direction) const override {
        auto cosine_theta = dot(unit_vector(direction), uvw.w());
        return fmax(0, cosine_theta/pi);
    }

    vec3 generate(metropolis_sampler& sampler) const override {
        return uvw.local(sampler.random_cosine_direction());
    }

  public:
    onb uvw;
};


class sphere_pdf : public pdf {
  public:
    sphere_pdf() { }

    double value(const vec3& direction) const override {
        return 1/ (4 * pi);
    }

    vec3 generate(metropolis_sampler& sampler) const override {
        return sampler.random_unit_vector();
    }
};


class hittable_pdf : public pdf {
  public:
    hittable_pdf(const hittable_list& _objects, const point3& _origin)
      : objects(_objects), origin(_origin)
    {}

    double value(const vec3& direction) const override {
        return objects.pdf_value(origin, direction);
    }

    vec3 generate(metropolis_sampler& sampler) const override {
        return objects.random(origin, sampler);
    }

  public:
    const hittable_list& objects;
    point3 origin;
};


class mixture_pdf : public pdf {
  public:
    mixture_pdf(shared_ptr<pdf> p0, shared_ptr<pdf> p1) {
        p[0] = p0;
        p[1] = p1;
    }

    double value(const vec3& direction) const override {
        return 0.5 * p[0]->value(direction) + 0.5 *p[1]->value(direction);
    }

    vec3 generate(metropolis_sampler& sampler) const override {
        if (sampler.get() < 0.5)
            return p[0]->generate(sampler);
        else
            return p[1]->generate(sampler);
    }

  public:
    shared_ptr<pdf> p[2];
};


#endif
