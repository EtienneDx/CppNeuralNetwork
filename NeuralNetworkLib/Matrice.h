#pragma once

class Matrice
{
private:
	float *mat_;
	int width_;
	int height_;
public:
	Matrice(int width, int height, bool randomize = false);
	Matrice(int length, float* values);
	Matrice(const Matrice& other);
	~Matrice();

	float& Get(size_t, size_t);
	const float Get(size_t, size_t) const;
	void Set(size_t i, size_t j, float value = 0);
	void Add(size_t i, size_t j, float value);

	void Dot(const Matrice & m, Matrice& ret) const;
	Matrice * Dot(const Matrice & m) const;

	void Hadamard(const Matrice & m);

	void Apply(float(*fct)(float current));
	Matrice * Transposed() const;
	const int GetWidth() const;
	const int GetHeight() const;

	void operator+=(const Matrice & m);
	void operator-=(const Matrice & m);
	Matrice * operator*(const float f) const;
	void operator=(const Matrice & m);

	friend std::ostream & operator << (std::ostream &out, const Matrice & mat);
};