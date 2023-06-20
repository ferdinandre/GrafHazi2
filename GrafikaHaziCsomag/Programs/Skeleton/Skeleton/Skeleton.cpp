//=============================================================================================
// 
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Ferdinand Andre Albert
// Neptun : G6MHH3
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"
const float epsilon = 0.0001f;
struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	Material(vec3 _kd, vec3 _ks, float _shininess) : ka(_kd* M_PI), kd(_kd), ks(_ks) { shininess = _shininess; }
};
float dist(vec3 a, vec3 b) {
	return dot(a - b, a - b);
}
struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};
struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};
class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};
class Triangle : public Intersectable {
public:
	vec3 a, b, c;
	vec3 n;
	Triangle() {

	}
	Triangle(vec3 _a, vec3 _b, vec3 _c) {

		a = _a; b = _b; c = _c;
		n = normalize(cross(b - a, c - a));

	}
	Hit intersect(const Ray& ray) {
		float t = dot(a - ray.start, n) / dot(ray.dir, n);
		vec3 p = ray.start + t * ray.dir;
		Hit hit;
		if (dot(cross(b - a, p - a), n) > 0 &&
			dot(cross(c - b, p - b), n) > 0 &&
			dot(cross(a - c, p - c), n) > 0) {
			hit.t = t; hit.normal = n; hit.position = p;

		}
		return hit;
	}

};
class Square : public Intersectable {
public:
	vec3 a, b, c, d;
	vec3 n;
	Square(vec3 _a, vec3 _b, vec3 _c, vec3 _d) {
		a = _a; b = _b; c = _c; d = _d;
		n = cross(c - b, d - b);
	
	}
	Square() { }
	Hit intersect(const Ray& ray) {
		Hit hit;
		float t = dot(a - ray.start, n) / dot(ray.dir, n);
		vec3 p = ray.start + t * ray.dir;

		if (dot(cross(b - a, p - a), n) > 0 &&
			dot(cross(c - b, p - b), n) > 0 &&
			dot(cross(d - c, p - c), n) > 0 &&
			dot(cross(a - d, p - d), n) > 0) {
			hit.position = p;
			hit.normal = n;
			hit.t = t;

		}
		return hit;
	}
};
class Octahedron : public Intersectable {
	Triangle sides[8];
public:
	Octahedron(vec3 refp, float sidelength) {
		vec3 vertices[6];
		vertices[0] = refp;
		float sqrt2_length = sidelength * sqrtf(2.0f); // gyök2-ször oldalhossz

		vertices[1] = refp + vec3(sqrt2_length * 0.5f, 0, sqrt2_length * 0.5f);
		vertices[2] = refp + vec3(0, sqrt2_length * 0.5f, sqrt2_length * 0.5f);
		vertices[3] = refp + vec3(-sqrt2_length * 0.5f, 0, sqrt2_length * 0.5f);
		vertices[4] = refp + vec3(0, -sqrt2_length * 0.5f, sqrt2_length * 0.5f);
		vertices[5] = refp + vec3(0, 0, sqrt2_length);

		sides[0] = Triangle(vertices[0], vertices[1], vertices[2]);
		sides[1] = Triangle(vertices[0], vertices[2], vertices[3]);
		sides[2] = Triangle(vertices[0], vertices[3], vertices[4]);
		sides[3] = Triangle(vertices[0], vertices[4], vertices[1]);

		sides[4] = Triangle(vertices[1], vertices[2], vertices[5]);
		sides[5] = Triangle(vertices[2], vertices[3], vertices[5]);
		sides[6] = Triangle(vertices[3], vertices[4], vertices[5]);
		sides[7] = Triangle(vertices[4], vertices[1], vertices[5]);

	}

	Hit intersect(const Ray& ray) {
		Hit bestHit;
		std::vector<Hit> hits;
		for (Triangle side : sides) {
			Hit currentHit = side.intersect(ray);
			if (currentHit.t > 0) {
				hits.push_back(currentHit);
			}
		}
		if (hits.size() > 0) {
			bestHit = hits.at(0);
			for (const auto& hit : hits) {
				if (hit.t < bestHit.t) {
					bestHit = hit;
				}
			}
		}
		return bestHit;
	}
};
class Cone : public Intersectable {
public:
	vec3 lightSource;
	vec3 p, n;
	vec3 color;
	float alpha;
	float height;
	Cone(vec3 _p, vec3 _n, float _a, float _h, vec3 _color) {
		p = _p, n = normalize(_n), alpha = _a; height = _h; color = _color;
		lightSource = p + n * epsilon * 50;
		
	}
	Hit intersect(const Ray& ray) {

		Hit hit;
		float t;
		float a = powf(dot(ray.dir, n), 2) - dot(ray.dir, ray.dir) * powf(cosf(alpha), 2);
		float b = dot(ray.dir, n) * dot(ray.start, n) - dot(ray.dir, n) * dot(p, n) + dot(ray.start, n) * dot(ray.dir, n)
			- dot(p, n) * dot(ray.dir, n)
			- powf(cosf(alpha), 2) * (dot(ray.dir, ray.start) - dot(ray.dir, p) + dot(ray.start, ray.dir) - dot(p, ray.dir));
		float c = powf(dot(ray.start, n), 2) - 2 * (dot(ray.start, n) * dot(p, n)) + powf(dot(p, n), 2)
			- powf(cosf(alpha), 2) * (dot(ray.start, ray.start) - 2 * dot(ray.start, p) + dot(p, p));
		float discr = b * b - 4 * a * c;
		float t1 = (-1.0f * b + sqrtf(discr)) / (2.0f * a);
		float t2 = (-1.0f * b - sqrtf(discr)) / (2.0f * a);

		Hit hit1; hit1.t = t1; hit1.position = ray.start + t1 * ray.dir;
		vec3 n1 = -normalize(2 * dot(hit1.position - p, n) * n - 2 * (hit1.position - p) * powf(cosf(alpha), 2));
		if (dot(n1, ray.dir) > 0) n1 = -n1;
		hit1.normal = n1;

		Hit hit2; hit2.t = t2; hit2.position = ray.start + t2 * ray.dir;
		vec3 n2 = -normalize(2 * dot(hit2.position - p, n) * n - 2 * (hit2.position - p) * powf(cosf(alpha), 2));
		if (dot(n2, ray.dir) > 0) n2 = -n2;
		hit2.normal = n2;

		bool hit1IsOn = 0 < dot(hit1.position - p, n) && dot(hit1.position - p, n) < height;
		bool hit2IsOn = 0 < dot(hit2.position - p, n) && dot(hit2.position - p, n) < height;

		if (hit1IsOn && hit2IsOn) {
			hit = hit1.t < hit2.t ? hit1 : hit2;
		}
		if ((hit1IsOn && !hit2IsOn) || (!hit1IsOn && hit2IsOn)) {
			hit = hit1IsOn ? hit1 : hit2;
		}

		return hit;

	}
};
class Tetrahedron : public Intersectable {
	Triangle sides[4];
public:
	Tetrahedron(vec3 refp, float sidelength) {
		vec3 vertices[4];
		vec3 sulypont = refp + vec3(0, sqrtf(3.0f) / 3.0f, 0) * sidelength;
		float sqrt32 = sqrtf(3.0f) / 2.0f; // gyök(3)/ 2
		float sqrt23 = sqrtf(2.0f) / sqrtf(3.0f);//gyök2 / gyök3
		vec3 midp = sulypont + vec3(0, 0, sidelength * 0.5f * sqrt23);

		vertices[0] = refp;
		vertices[1] = refp + sqrt32 * sidelength * vec3(0, 1, 0) + vec3(1, 0, 0) * sidelength * 0.5f;
		vertices[2] = refp + sqrt32 * sidelength * vec3(0, 1, 0) + vec3(-1, 0, 0) * sidelength * 0.5f;
		vertices[3] = sulypont + sqrt23 * sidelength * vec3(0, 0, 1);

		sides[0] = Triangle(vertices[0], vertices[1], vertices[2]);
		sides[1] = Triangle(vertices[0], vertices[1], vertices[3]);
		sides[2] = Triangle(vertices[0], vertices[2], vertices[3]);
		sides[3] = Triangle(vertices[1], vertices[2], vertices[3]);



	}
	Hit intersect(const Ray& ray) {
		Hit bestHit;
		std::vector<Hit> hits;
		for (Triangle side : sides) {
			Hit currentHit = side.intersect(ray);
			if (currentHit.t > 0) {
				hits.push_back(currentHit);
			}
		}
		if (hits.size() > 0) {
			bestHit = hits.at(0);
			for (const auto& hit : hits) {
				if (hit.t < bestHit.t) {
					bestHit = hit;
				}
			}
		}
		return bestHit;

	}
};
class Camera {
	vec3 eye, lookat, right, up;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
};
struct Light {
	vec3 direction;
	vec3 Le;
	Light(vec3 _direction, vec3 _Le) {
		direction = normalize(_direction);
		Le = _Le;
	}
};
float rnd() { return (float)rand() / RAND_MAX; }
Cone* c1 = new Cone(
	vec3(0.35f, 0.0f, 0.3f),
	vec3(0.0f, 1.0f, 0.0f),
	M_PI / 7.0f,
	0.12f,
	vec3(0.5f, 0.0f, 0.0f)

);
Cone* c2 = new Cone(
	vec3(0.0f, 0.1f, 0.4f),
	vec3(1.0f, 0.0f, 0.0f),
	M_PI / 7.0f,
	0.12f,
	vec3(0.0f, 0.5f, 0.0f)
);
Cone* c3 = new Cone(
	vec3(0.6f, 0.6f, 1.0f),
	vec3(0.0f, 0.0f, -1.0f),
	M_PI / 7.0f,
	0.12f,
	vec3(0.0f, 0.0f, 0.5f)
);
class Scene {
public:
	std::vector<Intersectable*> objects;
	std::vector<Cone*> lights;
	vec3 La = vec3(0.4f, 0.4f, 0.4f);
	Camera camera;
	void build() {
		vec3 eye = vec3(1.5f, 2.2f, 0.5f), vup = vec3(0, 0, 1), lookat = vec3(0.5, 0.5, 0.5);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);
		objects.push_back(
			new Square(
				vec3(0, 0, 0),
				vec3(1, 0, 0),
				vec3(1, 0, 1),
				vec3(0, 0, 1)
			)
		);
		objects.push_back(
			new Square(
				vec3(0, 0, 0),
				vec3(0, 1, 0),
				vec3(0, 1, 1),
				vec3(0, 0, 1)
			)
		);

		objects.push_back(
			new Square(
				vec3(0, 0, 0),
				vec3(1, 0, 0),
				vec3(1, 1, 0),
				vec3(0, 1, 0)
			)
		);
		objects.push_back(
			new Square(
				vec3(0, 0, 1),
				vec3(1, 0, 1),
				vec3(1, 1, 1),
				vec3(0, 1, 1)
			)
		);
		objects.push_back(
			new Tetrahedron(
				vec3(0.7f, 0.4f, 0.0f),
				0.4f
			)
		);
		objects.push_back(
			new Octahedron(
				vec3(0.4f, 0.7f, 0.0f),
				0.325f
			)
		);
		objects.push_back(c1);
		lights.push_back(c1);
		objects.push_back(c2);
		lights.push_back(c2);
		objects.push_back(c3);
		lights.push_back(c3);
	}
	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y), 1);
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}
	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = -bestHit.normal;
		return bestHit;
	}
	bool shadowIntersect_t_dist(Ray ray, float t) {
		for (Intersectable* object : objects)
			if (object->intersect(ray).t > 0 && object->intersect(ray).t <= t) return true;
		return false;
	}
	vec3 trace(Ray ray, int depth) {
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return vec3(0, 0, 0);
		vec3 outRadiance;
		float ambient = (0.2f * (1 + dot(normalize(hit.normal), normalize(ray.dir))));
		outRadiance = La + vec3(ambient, ambient, ambient);
		for (Cone* light : lights) {
			vec3 vizsgalt = hit.position + hit.normal * epsilon;

			//Fényforrásból vizsgált pontba
			vec3 egysegV = normalize(vizsgalt - light->lightSource);

			//Ugyanez megfordítva
			vec3 fenyFele = normalize(light->lightSource - vizsgalt);


			Ray rayToLight(vizsgalt, fenyFele);
			float t = (light->lightSource.x - vizsgalt.x) / fenyFele.x;
			if (dot(egysegV, light->n) > cosf(light->alpha)) {
				float dist = dot(vizsgalt - light->lightSource, vizsgalt - light->lightSource);
				if(!shadowIntersect_t_dist(rayToLight, t)) 
				outRadiance = outRadiance + (light->color / (t * t * 1.5f) );
			}

		}
		return outRadiance;
	}
};



GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char* vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

class FullScreenTexturedQuad {
	
public:
	unsigned int vao;	// vertex array object id and texture id
	Texture texture;
	
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image)
	{
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects
		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	}
	

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};
FullScreenTexturedQuad* fullScreenTexturedQuad;
// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();

	std::vector<vec4> image(windowWidth * windowHeight);
	scene.render(image);
	
	// copy image to GPU as a texture
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);
	

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}
// Window has become invalid: Redraw
void onDisplay() {
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();									// exchange the two buffers
}
// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
}
// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}
// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	Hit hit = scene.firstIntersect(scene.camera.getRay(pX, windowHeight - pY));
	Cone* closest;
	float dst = 10000000;
	for (Cone* light : scene.lights) {
		float currentDst = dist(light->p, hit.position);
		if (currentDst < dst) {
			dst = currentDst;
			closest = light;
			closest->p = hit.position + hit.normal * epsilon;
			closest->n = hit.normal;
			closest->lightSource = closest->p + closest->n * epsilon * 50;
		}
	}
	
	printf("Normal: %1.2f, %1.2f, %1.2f\n", hit.normal.x, hit.normal.y, hit.normal.z);
	
	std::vector<vec4> image(windowWidth * windowHeight);
	scene.render(image);
	delete fullScreenTexturedQuad;
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);
	fullScreenTexturedQuad->Draw();
}
// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}
// Idle event indicating that some time elapsed: do animation here
void onIdle() {
}