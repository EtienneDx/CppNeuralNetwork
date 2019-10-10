#include "stdafx.h"
#include "Matrice.h"


Matrice::Matrice(int width, int height, bool randomize) : width_(width), height_(height)
{
	this->mat_ = new float[width * height]();
	if (randomize)
	{
		for (size_t i = 0; i < width * height; i++)
		{
			this->mat_[i] = (2.0 * std::rand()) / RAND_MAX - 1;
		}
	}
}

Matrice::Matrice(int length, float * values) : width_(length), height_(1)
{
	this->mat_ = new float[length]();
	for (size_t i = 0; i < length; i++)
	{
		this->mat_[i] = values[i];
	}
}

Matrice::Matrice(const Matrice & other) : width_(other.GetWidth()), height_(other.GetHeight())
{
	this->mat_ = new float[this->width_ * this->height_]();
	for (size_t i = 0; i < this->width_ * this->height_; i++)
	{
		this->mat_[i] = other.mat_[i];
	}
}

Matrice::~Matrice()
{
	delete [] this->mat_;
}

float& Matrice::Get(size_t i, size_t j)
{
	return this->mat_[i * this->GetWidth() + j];
}

const float Matrice::Get(size_t i, size_t j) const
{
	return this->mat_[i * this->GetWidth() + j];
}

void Matrice::Set(size_t i, size_t j, float value)
{
	this->mat_[i * this->width_ + j] = value;
}

void Matrice::Add(size_t i, size_t j, float value)
{
	this->mat_[i * this->width_ + j] += value;
}

void Matrice::Dot(const Matrice & m, Matrice & ret) const
{
	if (this->GetWidth() != m.GetHeight())
	{
		throw "Invalid matrices sizes! First matrice width must be equal to the second's height!";
	}
	if (this->GetHeight() != ret.GetHeight() || m.GetWidth() != ret.GetWidth())
	{
		throw "Invalid return matrice size, it should match the result from the dot product";
	}

	for (size_t i = 0; i < ret.GetHeight(); i++)// for each row
	{
		for (size_t j = 0; j < ret.GetWidth(); j++)// for each col
		{
			ret.Set(i, j);
			for (size_t k = 0; k < this->GetWidth(); k++)// for each col in start mat
			{
				float f = this->Get(i, k) * m.Get(k, j);
				ret.Add(i, j, f);
			}
		}
	}
}
Matrice Matrice::Dot(const Matrice & m) const
{
	Matrice ret(m.GetWidth(), this->GetHeight());
	this->Dot(m, ret);

	return ret;
}

void Matrice::Hadamard(const Matrice & m)
{
	if (this->GetHeight() != m.GetHeight() || m.GetWidth() != this->GetWidth())
	{
		throw "Invalid return matrice size, both matrices should have the same dimension";
	}
	for (size_t i = 0; i < this->GetHeight(); i++)// for each row
	{
		for (size_t j = 0; j < this->GetWidth(); j++)// for each col
		{
			this->Get(i, j) *= m.Get(i, j);
		}
	}
}

void Matrice::Apply(float(*fct)(float current))
{
	for (size_t i = 0; i < this->GetWidth() * this->GetHeight(); i++)
	{
		this->mat_[i] = (*fct)(this->mat_[i]);
	}
}

Matrice Matrice::Transposed() const
{
	Matrice mat(this->GetHeight(), this->GetWidth());
	for (size_t i = 0; i < this->GetHeight(); i++)
	{
		for (size_t j = 0; j < this->GetWidth(); j++)
		{
			mat.Set(j, i, this->Get(i, j));
		}
	}
	return mat;
}

const int Matrice::GetWidth() const
{
	return this->width_;
}

const int Matrice::GetHeight() const
{
	return this->height_;
}

void Matrice::operator+=(const Matrice & m)
{
	for (size_t i = 0; i < this->GetHeight(); i++)
	{
		for (size_t j = 0; j < this->GetWidth(); j++)
		{
			this->Add(i, j, m.Get(i, j));
		}
	}
}

void Matrice::operator-=(const Matrice & m)
{
	for (size_t i = 0; i < this->GetHeight(); i++)
	{
		for (size_t j = 0; j < this->GetWidth(); j++)
		{
			this->Add(i, j, -m.Get(i, j));
		}
	}
}

Matrice Matrice::operator*(const float f) const
{
	Matrice ret(*this);
	for (size_t i = 0; i < this->GetHeight() * this->GetWidth(); i++)
	{
		ret.mat_[i] *= f;
	}
	return ret;
}

void Matrice::operator=(const Matrice & m)
{
	delete[] this->mat_;
	this->width_ = m.GetWidth();
	this->height_ = m.GetHeight();
	this->mat_ = new float[this->width_ * this->height_]();
	for (size_t i = 0; i < this->width_ * this->height_; i++)
	{
		this->mat_[i] = m.mat_[i];
	}
}

float * Matrice::operator[](int i) const
{
	return this->mat_ + (i * this->GetWidth());
}

std::ostream & operator<<(std::ostream & out, const Matrice & mat)
{
	out << "[ ";
	for (size_t i = 0; i < mat.GetHeight(); i++)
	{
		if(i)
			out << std::endl << "  ";
		for (size_t j = 0; j < mat.GetWidth(); j++)
		{
			out << mat.Get(i, j) << "\t";
		}
	}
	out << " ]";
	return out;
}
